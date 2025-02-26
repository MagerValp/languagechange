import torch
import warnings
import json
import argparse
from peft import PeftModel
from datasets import Dataset
from huggingface_hub import login
from typing import Literal, Sequence, TypedDict
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define types for chat dialog outside the class for clarity
Role = Literal["system", "user"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class DefinitionGenerator:
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
                 batch_size: int = 32, max_time: float = 4.5):
        """
        Sets up the DefinitionGenerator with model and data details.

        Args:
            model_name: The base model name from Hugging Face (e.g., "meta-llama/Llama-2-7b-chat-hf").
            ft_model_name: The fine-tuned model name (e.g., "FrancescoPeriti/Llama2Dictionary").
            hf_token: Hugging Face token to access models.
            testdata_path: Path to a JSON file with test examples.
            batch_size: Number of examples to process in one go (default is 32).
            max_time: Max time in seconds per batch (default is 4.5).

        Example:
            gen = DefinitionGenerator("meta-llama/Llama-2-7b-chat-hf", 
                                       "FrancescoPeriti/Llama2Dictionary", 
                                       "hf_token", "testdata.json")
        """
        self.model_name = model_name
        self.ft_model_name = ft_model_name
        self.hf_token = hf_token
        self.testdata_path = testdata_path
        self.batch_size = batch_size
        self.max_time = max_time

        # Log in to Hugging Face with token
        login(self.hf_token)

        # Load the base model and apply the fine-tuned version
        chat_model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto')
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
            FileNotFoundError: If the JSON file doesn’t exist.
            json.JSONDecodeError: If the JSON file is badly formatted.

        Example:
            JSON file content: [{"target": "run", "example": "I run every morning."}]
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
        Turns the prompts into a format the model can understand (tokens) with a fixed length.

        Uses a max length of 512 tokens and pads shorter inputs to match.
        """
        max_length = 512

        def formatting_func(record):
            return record['text']

        def tokenization_func(dataset):
            return self.tokenizer(
                formatting_func(dataset),
                truncation=True,
                max_length=max_length,
                padding="max_length",
                add_special_tokens=False
            )

        self.tokenized_dataset = self.dataset.map(tokenization_func)

    def extract_definition(self, answer: str) -> str:
        """
        Pulls out the actual definition from the model’s response, depending on the model type.

        Args:
            answer: The text generated by the model.

        Returns:
            A cleaned-up definition string, with newlines replaced by spaces and a newline at the end.

        Notes:
            - For Llama-2, looks for text after '[/INST]'.
            - For Llama-3, takes the last line.
            - Warns if something looks off (e.g., empty output).
        """
        if "Llama-2" in self.model_name:
            definition = answer.split('[/INST]')[-1].strip(" .,;:")
            if 'SYS>>' in definition:
                definition = ''
                warnings.warn("The output looks wrong—maybe the example is too long. Try shortening it.")
        elif "Llama-3" in self.model_name:
            definition = answer.split('\n')[-1].strip(" .,;:")
            if not definition:
                warnings.warn("Got an empty definition—check if the example is too long.")
        else:
            definition = ''
            warnings.warn("This model isn’t supported yet—add logic in extract_definition if needed.")
        return definition.replace('\n', ' ') + '\n'

    def generate_definitions(self):
        """
        Uses the model to generate definitions for all examples in batches.

        Adds the definitions to the dataset under a new 'definition' column.
        """
        sense_definitions = []
        with torch.no_grad():
            for i in range(0, len(self.tokenized_dataset), self.batch_size):
                batch = self.tokenized_dataset[i:i + self.batch_size]

                model_input = {k: torch.tensor(batch[k]).to('cuda') for k in ['input_ids', 'attention_mask']}

                try:
                    output_ids = self.model.generate(
                        **model_input,
                        max_length=512,
                        forced_eos_token_id=self.eos_tokens,
                        max_time=self.max_time * self.batch_size,
                        eos_token_id=self.eos_tokens,
                        temperature=0.00001,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                except Exception as e:
                    warnings.warn(f"Failed to generate for batch {i // self.batch_size}: {e}")
                    sense_definitions.extend([''] * len(batch))
                    continue

                answers = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                sense_definitions.extend(self.extract_definition(answer) for answer in answers)

        self.dataset = self.dataset.add_column('definition', sense_definitions)

    def print_results(self):
        """
        Displays the target word, example sentence, and generated definition for each entry.
        """
        for row in self.dataset:
            print(f"Target: {row['target']}\nExample: {row['example']}\nSense definition: {row['definition']}")

    def run(self):
        """
        Runs the whole process from loading data to printing results in one go.
        """
        self.load_dataset()
        self.apply_chat_template()
        self.tokenization()
        self.generate_definitions()
        self.print_results()

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate word definitions using fine-tuned Llama models.")
    parser.add_argument("--model", type=str, required=True, help="Base model name (e.g., meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--ft_model", type=str, required=True, help="Fine-tuned model name (e.g., FrancescoPeriti/Llama2Dictionary)")
    parser.add_argument("--testdata", type=str, required=True, help="Path to JSON test data file")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of examples per batch (default: 32)")
    parser.add_argument("--max_time", type=float, default=4.5, help="Max time per batch in seconds (default: 4.5)")

    args = parser.parse_args()

    # Create and run the generator
    generator = DefinitionGenerator(
        model_name=args.model,
        ft_model_name=args.ft_model,
        hf_token=args.hf_token,
        testdata_path=args.testdata,
        batch_size=args.batch_size,
        max_time=args.max_time
    )
    generator.run()