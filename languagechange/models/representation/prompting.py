from typing import Tuple, List, Union, Any
from languagechange.usages import TargetUsage
import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import logging
import trankit


class SCFloat(BaseModel):
    change : float = Field(description='The semantic change on a scale from 0 to 1.',le=1, ge=0)


class SCDURel(BaseModel):
    change : int = Field(description='The semantic similary from 1 to 4, where 1 is unrelated, 2 is distantly related, 3 is closely related and 4 is identical.',le=4, ge=1)


class PromptModel:
    def __init__(self, model_name : str, model_provider : str, langsmith_key : str = None, provider_key_name : str = None, provider_key : str = None, scale="float"):
        self.model_name = model_name

        os.environ["LANGSMITH_TRACING"] = "true"
        
        # The keys can either be passed as arguments, stored as an environment variable or put in manually
        if langsmith_key != None:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
        elif not os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")

        if provider_key_name is None:
            provider_key_name = {'openai':"OPENAI_API_KEY", "groq":"GROQ_API_KEY"}[model_provider] # add more providers
        if provider_key != None:
            os.environ[provider_key_name] = provider_key
        elif not os.environ.get(provider_key_name):
            os.environ[provider_key_name] = getpass.getpass(f"Enter API key for {model_provider}: ")

        llm = init_chat_model(model_name, model_provider=model_provider)

        if scale == "float":
            self.structure = SCFloat
        elif scale == "DURel":
            self.structure = SCDURel
        else:
            self.structure = None

        if self.structure != None:
            self.model = llm.with_structured_output(self.structure)
        else:
            self.model = llm


    def get_response(self, target_usages : List[TargetUsage], 
                     system_message = 'You are a lexicographer',
                     user_prompt_template = 'Please provide a number measuring how different the meaning of the word \'{target}\' is between the following example sentences: \n1. {usage_1}\n2. {usage_2}',
                     lemmatize = True):
        
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
                
        if lemmatize:
            p = trankit.Pipeline("english")
            lemmatized = [p.lemmatize(sentence, is_sent = True) for sentence in sentences]
            lemmas = [get_lemma(lemmatized[i], target_usages[i]) for i in range(2)]
            
            if lemmas[0] != lemmas[1]:
                logging.info("Lemmas of the two words differ, are you sure they are the same?")
            target = lemmas[0]
        else:
            target = words[0]

        prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", user_prompt_template)]
        )
            
        prompt = prompt_template.invoke({"target": target, "usage_1": sentences[0], "usage_2": sentences[1]})
        
        try:
            response = self.model.invoke(prompt)
        except:
            logging.info("Could not run chat completion.")
            return None
        
        try:
            return response.change
        except:
            return response