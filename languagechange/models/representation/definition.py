import os
import torch
import warnings
import json
import argparse
from peft import PeftModel
from datasets import Dataset
from huggingface_hub import login
from typing import Literal, Sequence, TypedDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from languagechange.usages import TargetUsage
from pydantic import BaseModel, Field
import logging
from typing import Tuple, List, Union, Any

# Define types for chat dialog outside the class for clarity
Role = Literal["system", "user"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class LlamaDefinitionGenerator:
    """
    A tool to create short, clear definitions for words based on example sentences using fine-tuned Llama models.

    Attributes:
        model_name (str): Name of the base model (e.g., "meta-llama/Llama-2-7b-chat-hf").
        ft_model_name (str): Name of the fine-tuned model (e.g., "FrancescoPeriti/Llama2Dictionary").
        hf_token (str): Hugging Face token for authentication.
        testdata_path (str): Path to JSON file with test data.
        batch_size (int): How many examples to process at once.
        max_time (float): Max time (in seconds) allowed per batch.
        model: The loaded fine-tuned model ready for generation.
        tokenizer: The tokenizer that prepares text for the model.
        eos_tokens: Tokens that tell the model when to stop generating.
        dataset: data loaded from the JSON file.
        tokenized_dataset: data after being tokenized for the model.
    """

    def __init__(self, model_name: str, ft_model_name: str, hf_token: str, testdata_path: str, 
                 batch_size: int = 32, max_time: float = 4.5, max_length: int = 512, temperature: float = 0.7):
        """
        Sets up the LlamaDefinitionGenerator with model and data details.

        Args:
            model_name: The base model name from Hugging Face (e.g., "meta-llama/Llama-2-7b-chat-hf").
            ft_model_name: The fine-tuned model name (e.g., "FrancescoPeriti/Llama2Dictionary").
            hf_token: Hugging Face token to access models.
            testdata_path: Path to a JSON file with test examples.
            batch_size: Number of examples to process in one go (default is 32).
            max_time: Max time in seconds per batch (default is 4.5).
            
        Example:
            gen = LlamaDefinitionGenerator("meta-llama/Llama-2-7b-chat-hf", 
                                       "FrancescoPeriti/Llama2Dictionary", 
                                       "hf_token", "testdata.json")
        """
        self.model_name = model_name
        self.ft_model_name = ft_model_name
        self.hf_token = hf_token
        self.testdata_path = testdata_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_time = max_time
        self.temperature = temperature

        # Log in to Hugging Face with token
        login(self.hf_token)

        # Load the base model with explicit settings
        chat_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
                    low_cpu_mem_usage=True
                )
        self.model = PeftModel.from_pretrained(chat_model, self.ft_model_name)
        self.model.eval()

        # Set up the tokenizer with Llama-specific settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define tokens that signal the end of a definition
        self.eos_tokens = [self.tokenizer.encode(token, add_special_tokens=False)[0] 
                           for token in [';', ' ;', '.', ' .']]
        self.eos_tokens.append(self.tokenizer.eos_token_id)

    def load_dataset(self):
        """
        Loads test data from a JSON file into a dataset.

        The JSON should be a list of objects, each with 'target' (the word) and 'example' (the sentence).

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            json.JSONDecodeError: If the JSON file is badly formatted.
        """
        try:
            with open(self.testdata_path, 'r') as f:
                examples = json.load(f)
            self.dataset = Dataset.from_list(examples)
        except FileNotFoundError:
            raise FileNotFoundError(f"Couldn’t find the test data file at: {self.testdata_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"The JSON file at {self.testdata_path} isn’t valid.")

    def apply_chat_template(self):
        """
        Creates prompts for the model by combining a system message and user question for each example.

        The system message sets the model as a lexicographer, and the user message asks for a definition.
        """
        system_message = "You are a lexicographer familiar with providing concise definitions of word meanings."
        template = 'Please provide a concise definition for the meaning of the word "{}" in the following sentence: {}'

        def apply_chat_template_func(record):
            dialog = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": template.format(record['target'], record['example'])}
            ]
            prompt = self.tokenizer.decode(self.tokenizer.apply_chat_template(dialog, add_generation_prompt=True))
            return {'text': prompt}

        self.dataset = self.dataset.map(apply_chat_template_func)

    def tokenization(self):
        """
        Tokenizes the prompts into model-ready format with fixed length of 512 tokens.

        Pads shorter inputs and truncates longer ones.
        """

        def formatting_func(record):
            return record['text']

        def tokenization_func(dataset):
            return self.tokenizer(
                formatting_func(dataset),
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                add_special_tokens=False
            )

        self.tokenized_dataset = self.dataset.map(tokenization_func)

    def extract_definition(self, answer: str) -> str:
        """
        Extracts the actual definition from the model's response based on model type.

        Args:
            answer: The text generated by the model.

        Returns:
            A cleaned-up definition string with proper formatting.

        Notes:
            - For Llama-2: Extracts text after '[/INST]'
            - For Llama-3: Takes the last line
            - Warns if output appears abnormal
        """
        if "Llama-2" in self.model_name:
            definition = answer.split('[/INST]')[-1].strip(" .,;:")
            if 'SYS>>' in definition:
                definition = ''
                warnings.warn("Abnormal output detected - the example sentence might be too long. Try shortening it.")
        elif "Llama-3" in self.model_name:
            definition = answer.split('\n')[-1].strip(" .,;:")
            if not definition:
                warnings.warn("Empty definition generated - check if example sentence is too long.")
        else:
            definition = ''
            warnings.warn("Model type not supported - add handling logic in extract_definition if needed")
        return definition.replace('\n', ' ') + '\n'

    def generate_definitions(self):
        """
        Generates definitions for all examples in batches using the model.

        Adds the definitions to the dataset in a new 'definition' column.
        """
        sense_definitions = []
        device = next(self.model.parameters()).device  # Get model's device
        
        print(f"Using generation device: {device}")
        print(f"EOS tokens: {self.eos_tokens}")

        with torch.no_grad():
            for i in range(0, len(self.tokenized_dataset), self.batch_size):
                batch = self.tokenized_dataset[i:i + self.batch_size]

                # Convert to tensors and move to model's device
                input_ids = torch.tensor(batch['input_ids']).to(device)
                attention_mask = torch.tensor(batch['attention_mask']).to(device)

                model_input = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

                try:
                    output_ids = self.model.generate(
                        **model_input,
                        max_new_tokens=50,  # Generate up to 50 new tokens
                        forced_eos_token_id=self.eos_tokens,
                        max_time=self.max_time * self.batch_size,
                        eos_token_id=self.eos_tokens,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    answers = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    sense_definitions.extend(self.extract_definition(answer) for answer in answers)
                except Exception as e:
                    warnings.warn(f"Failed to generate for batch {i//self.batch_size}: {e}")
                    current_batch_size = len(batch['input_ids'])
                    sense_definitions.extend([''] * current_batch_size)
                    continue

        # Validate output length
        if len(sense_definitions) != len(self.dataset):
            raise ValueError(f"Generated definitions count ({len(sense_definitions)}) "
                             f"doesn't match dataset size ({len(self.dataset)})")

        self.dataset = self.dataset.add_column('definition', sense_definitions)
    
    

    def print_results(self):
        """
        Displays the target word, example sentence, and generated definition for each entry.
        """
        for row in self.dataset:
            print(f"Target: {row['target']}\nExample: {row['example']}\nDefinition: {row['definition']}")

    def run(self):
        """
        Executes the complete workflow from data loading to result printing.
        """
        self.load_dataset()
        self.apply_chat_template()
        self.tokenization()
        self.generate_definitions()
        self.print_results()




# Data model representing the definition of a target word within an example sentence.
class DefinitionOutput(BaseModel):
    """
    Represents the structured output for a word definition.

    Attributes:
        target (str): The target word.
        example (str): The example sentence.
        definition (str): The concise definition of the target word as used in the sentence.
    """
    target: str = Field(description="The target word")
    example: str = Field(description="The example sentence")
    definition: str = Field(description="The definition of the target word as used in the sentence")

class ChatModelDefinitionGenerator:
    """
    A model to generate concise definitions for target words using a chat model with structured output.

    The model leverages an underlying chat model (initialized with LangChain) that returns a 
    structured DefinitionOutput.
    """
    def __init__(self, model_name: str, model_provider: str, 
                 langsmith_key: str = None, provider_key_name: str = None, 
                 provider_key: str = None, language: str = None):
        """
        Initializes the DefinitionModel.

        Args:
            model_name (str): The name of the model.
            model_provider (str): The model provider (e.g., "openai").
            langsmith_key (str, optional): API key for LangSmith. Defaults to None.
            provider_key_name (str, optional): Environment variable name for the provider API key. Defaults to None.
            provider_key (str, optional): The provider API key. Defaults to None.
            language (str, optional): Language code for potential lemmatization. Defaults to None.
        """
        self.model_name = model_name
        self.language = language

        os.environ["LANGSMITH_TRACING"] = "true"

        # Set the LangSmith API key.
        if langsmith_key is not None:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
        elif not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")

        # Determine the provider key name if not provided.
        if provider_key_name is None:
            provider_key_names = {
                "openai": "OPENAI_API_KEY",
                "groq": "GROQ_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "cohere": "COHERE_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
                "fireworks": "FIREWORKS_API_KEY",
                "mistralai": "MISTRAL_API_KEY",
                "together": "TOGETHER_API_KEY",
                "xai": "XAI_API_KEY"
            }
            if model_provider in provider_key_names.keys():
                provider_key_name = provider_key_names[model_provider]
        # Set the provider API key.
        if provider_key is not None:
            os.environ[provider_key_name] = provider_key
        elif not os.environ.get(provider_key_name):
            os.environ[provider_key_name] = getpass.getpass(f"Enter API key for {model_provider}: ")

        try:
            llm = init_chat_model(model_name, model_provider=model_provider)
        except Exception as e:
            logging.error("Could not initialize chat model.")
            raise e

        # Configure the model to use structured output with the DefinitionOutput schema.
        self.model = llm.with_structured_output(DefinitionOutput)

    def get_definitions(self, target_usages: List) -> List[str]:
        """
        Generates concise definitions for each target usage provided.

        Each target usage is expected to have either:
          - 'target' and 'example' attributes, or 
          - a text with offsets indicating the target word location.

        Args:
            target_usages (List): A list of target usage instances.

        Returns:
            List[str]: A list of definitions or full responses if structured output fails.
        """
        definitions = []
        for usage in target_usages:
            # Use the provided attributes if available; otherwise, extract from text and offsets.
            if hasattr(usage, 'target') and hasattr(usage, 'example'):
                target_word = usage.target
                example_sentence = usage.example
            else:
                target_word = usage.text()[usage.offsets[0]:usage.offsets[1]]
                example_sentence = usage.text()
            
            system_message = "You are a lexicographer familiar with providing concise definitions of word meanings."
            user_prompt_template = ("Please provide a concise definition for the meaning of the word '{target}' "
                                    "as used in the following sentence:\nSentence: {example}")
            
            prompt_template = ChatPromptTemplate.from_messages(
                [("system", system_message), ("user", user_prompt_template)]
            )
            
            prompt = prompt_template.invoke({"target": target_word, "example": example_sentence})
            
            try:
                response = self.model.invoke(prompt)
            except Exception as e:
                logging.error("Could not run chat completion.")
                raise e
            
            try:
                definitions.append(response.definition)
            except Exception:
                definitions.append(response)
        
        return definitions