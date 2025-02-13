from typing import Tuple, List, Union, Any
from languagechange.usages import TargetUsage
from openai import OpenAI
import logging
import re
import trankit

# Generative model, should be compatible both with ChatGPT and DeepInfra.
class PromptModel:
    def __init__(self, model_name : str, key, base_url : str = None):
        self.model_name = model_name
        if not base_url:
            self.client = OpenAI(api_key = key)
        else:
            self.client = OpenAI(api_key = key, base_url=base_url) # For DeepInfra use "https://api.deepinfra.com/v1/openai"


    def get_response(self, target_usages : List[TargetUsage], 
                     system_message = 'You are a lexicographer',
                     prompt_template = 'Please provide a single number between 0 and 1 measuring how different the meaning of the word \'{}\' is between the following example sentences: \n1. {}\n2. {}'):
        
        assert len(target_usages) == 2

        words = []
        sentences = []
        for usage in target_usages:
            words.append(usage.text()[usage.offsets[0]:usage.offsets[1]])
            sentences.append(usage.text())

        def get_lemma(tokenized, usage):
            for token in tokenized['tokens']:
                if token['span'] == tuple(usage.offsets):
                    return(token['lemma'])
                
        p = trankit.Pipeline("english")
        lemmatized = [p.lemmatize(sentence, is_sent = True) for sentence in sentences]
        lemmas = [get_lemma(lemmatized[i], target_usages[i]) for i in range(2)]
        
        if lemmas[0] != lemmas[1]:
            logging.info("Lemmas of the two words differ, are you sure they are the same?")
        lemma = lemmas[0]
            
        prompt = prompt_template.format(lemma, sentences[0], sentences[1])

        logging.info(f"Prompt: {prompt}")
        print(prompt)
        
        try:
            completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "developer", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            )

            message = completion.choices[0].message()
        except:
            logging.info("Could not run chat completion.")
            return None

        # Extract the numbers from the answer of the chat model
        numbers = []
        for match in re.finditer('\s\d(\.\d+)?',message):
            number = message[match.start():match.end()]
            try:
                if 0 <= float(number) <= 1:
                    numbers.append(float(number))
            except:
                continue
        # Assuming the model's answer will be the first number it mentions
        return numbers[0]
        
