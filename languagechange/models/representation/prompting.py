from typing import Tuple, List, Union, Any
from languagechange.usages import TargetUsage
from openai import OpenAI
import logging


# Generative model, should be compatible both with ChatGPT and DeepInfra.
class PromptModel:
    def __init__(self, model_name : str, key, base_url : str = ''):
        self.model_name = model_name
        if base_url == '':
            self.client = OpenAI(api_key = key)
        else:
            self.client = OpenAI(api_key = key, base_url=base_url) # For DeepInfra use "https://api.deepinfra.com/v1/openai"


    def get_response(self, target_usages : List[TargetUsage], 
                     system_message = 'You are a lexicographer',
                     prompt_template = 'Please provide a single number between 0 and 1 measuring how different the meaning of the word {} is between the following example sentences: \n1. {}\n2. {}'):
        
        assert len(target_usages) == 2

        words = []
        sentences = []
        for usage in target_usages:
            words.append(usage.text()[usage.offsets[0]:usage.offsets[1]])
            sentences.append(usage.text())
        
        # todo: Ideally use some lemmatization function to use the base form of the target word instead of one of the occurrences
        prompt = prompt_template.format(words[0], sentences[0], sentences[1])
        
        try:
            completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            )

            return completion.choices[0].message()
        except:
            logging.info("Could not run chat completion.")