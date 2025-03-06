import warnings
import json
import argparse
from datasets import Dataset
from typing import Literal, Sequence, TypedDict
import openai
import time

# Define types for chat messages
Role = Literal["system", "user"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class ChatGPTDefinitionGenerator:
    """
    A simple tool to make short definitions for words using ChatGPT, based on example sentences.

    Attributes:
        openai_api_key (str): Your OpenAI API key.
        testdata_path (str): The path to the JSON file with test data.
        batch_size (int): How many examples to process at one time.
        request_delay (float): Time to wait between API requests (in seconds).
        model (str): The ChatGPT model to use, like "gpt-3.5-turbo".
        dataset: The data loaded from the JSON file.
    """

    def __init__(self, openai_api_key: str, testdata_path: str, batch_size: int = 32, 
                 request_delay: float = 1.0, model: str = "gpt-3.5-turbo"):
        """
        Sets up the tool with your API key and data details.

        Args:
            openai_api_key: Your OpenAI API key.
            testdata_path: The path to the JSON file with examples.
            batch_size: Number of examples to process at once (default is 32).
            request_delay: Time to wait between requests (default is 1 second).
            model: The ChatGPT model to use (default is "gpt-3.5-turbo").
        """
        self.openai_api_key = openai_api_key
        self.testdata_path = testdata_path
        self.batch_size = batch_size
        self.request_delay = request_delay
        self.model = model
        # Set the API key for OpenAI to use
        openai.api_key = self.openai_api_key

    def load_dataset(self):
        """
        Loads the test data from a JSON file.

        The JSON file should have a list of items, each with a 'target' word and an 'example' sentence.

        Raises:
            FileNotFoundError: If the file can't be found.
            json.JSONDecodeError: If the JSON file has a bad format.
        """
        try:
            with open(self.testdata_path, 'r') as f:
                examples = json.load(f)
            self.dataset = Dataset.from_list(examples)
        except FileNotFoundError:
            raise FileNotFoundError(f"Can't find the test data file at: {self.testdata_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"The JSON file at {self.testdata_path} is not right.")

    def apply_chat_template(self):
        """
        Makes messages for ChatGPT with a system message and a user question for each example.

        The system message tells ChatGPT what to do, and the user question asks for a definition.
        """
        system_message = "You are a lexicographer familiar with providing concise definitions of word meanings."
        template = 'Please provide a concise definition for the meaning of the word "{}" in the following sentence: {}'

        def apply_chat_template_func(record):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": template.format(record['target'], record['example'])}
            ]
            return {'messages': messages}

        self.dataset = self.dataset.map(apply_chat_template_func)

    def generate_definitions(self):
        """
        Uses ChatGPT to make definitions for all examples.

        Adds the definitions to the dataset in a new 'definition' column.
        """
        definitions = []
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]
            for messages in batch['messages']:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=50,
                        temperature=0.7
                    )
                    definition = response.choices[0].message.content.strip()
                    definitions.append(definition)
                except Exception as e:
                    warnings.warn(f"Could not make definition: {e}")
                    definitions.append('')
                time.sleep(self.request_delay)
        self.dataset = self.dataset.add_column('definition', definitions)

    def print_results(self):
        """
        Shows the target word, example sentence, and definition for each item.
        """
        for row in self.dataset:
            print(f"Target: {row['target']}\nExample: {row['example']}\nDefinition: {row['definition']}\n")

    def run(self):
        """
        Runs the whole process from loading data to showing results.
        """
        self.load_dataset()
        self.apply_chat_template()
        self.generate_definitions()
        self.print_results()

if __name__ == "__main__":
    # Set up command-line options
    parser = argparse.ArgumentParser(description="Make word definitions with ChatGPT.")
    parser.add_argument("--openai_api_key", type=str, required=True, help="Your OpenAI API key")
    parser.add_argument("--testdata", type=str, required=True, help="Path to the JSON test data file")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of examples per batch (default: 32)")
    parser.add_argument("--request_delay", type=float, default=1.0, help="Delay between requests in seconds (default: 1.0)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="ChatGPT model to use (default: gpt-3.5-turbo)")

    args = parser.parse_args()

    # Create and start the tool
    generator = ChatGPTDefinitionGenerator(
        openai_api_key=args.openai_api_key,
        testdata_path=args.testdata,
        batch_size=args.batch_size,
        request_delay=args.request_delay,
        model=args.model
    )
    generator.run()