import os
import json
import argparse
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class DefinitionOutput(BaseModel):
    """
    Data model representing the definition of a target word within an example sentence.
    """
    target: str = Field(description="The target word")
    example: str = Field(description="The example sentence")
    definition: str = Field(description="The definition of the target word as used in the sentence")

class ChatGPTDefinitionGenerator:
    """
    A class to generate definitions for target words based on provided example sentences using LangChain and ChatGPT.
    """
    def __init__(self, openai_api_key: str, testdata_path: str):
        """
        Initializes the DefinitionGenerator with the OpenAI API key and test data file path.
        
        Args:
            openai_api_key (str): The API key for OpenAI.
            testdata_path (str): The file path to the JSON test data.
        """
        # Set the OpenAI API key as an environment variable
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Load test data from the JSON file
        with open(testdata_path, "r") as f:
            self.input_data = json.load(f)

        # Define the prompt template for generating definitions
        self.prompt_template = PromptTemplate(
            input_variables=["target", "example"],
            template="You are a lexicographer familiar with providing concise definitions of word meanings. Please provide a concise definition for the meaning of the word '{target}' and the sentencein the following sentence: '{example}'."
        )
        
        # Initialize the ChatOpenAI model with a specified temperature
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Create the chain by combining the prompt template with the LLM model configured for structured output
        self.chain = self.prompt_template | self.llm.with_structured_output(DefinitionOutput)

    def generate_definitions(self) -> list:
        """
        Generates definitions for each target word in the input data and returns a list of JSON strings.
        
        Returns:
            list: A list of JSON strings representing the generated definitions.
        """
        definitions = []
        for item in self.input_data:
            target = item["target"]
            example = item["example"]
            # Generate structured output using the chain
            structured_output = self.chain.invoke({"target": target, "example": example})
            # print('Test: ', structured_output)
            # Convert the structured output to a JSON string
            definition_json = structured_output.model_dump_json()
            definitions.append(definition_json)
        return definitions

def main():
    """
    Main function to parse command-line arguments and generate definitions.
    """
    parser = argparse.ArgumentParser(description="Definition Generator using LangChain and ChatGPT")
    parser.add_argument("--openai_api_key", required=True, help="OpenAI API Key")
    parser.add_argument("--testdata", required=True, help="Path to the test data JSON file")
    args = parser.parse_args()

    # Create an instance of DefinitionGenerator with provided arguments
    generator = ChatGPTDefinitionGenerator(args.openai_api_key, args.testdata)

    # Generate definitions and retrieve a list of JSON strings
    results = generator.generate_definitions()

    # Print each JSON string
    for result in results:
        print(result)


if __name__ == "__main__":
    main()