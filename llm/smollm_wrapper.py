import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import torch # type: ignore
from torch import compile # type: ignore
from peft import PeftModel # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
from langchain_huggingface import HuggingFacePipeline # type: ignore
from langchain_core.runnables import Runnable # type: ignore

import warnings
warnings.filterwarnings('ignore')

class SmolLLMWrapper:
    def __init__(self, base_model_id, model_id, max_length=512, temperature=0.3, top_p=0.9, top_k=50,
                         repetition_penalty=1.2, do_sample=True, truncation=True):
        
        print("Initializing LLM Model")
        print("-"*20)
        # device initialization
        self.my_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Base model initialization
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token or self.base_tokenizer.unk_token
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16).to(self.my_device)

        # Pre-trained model initialization
        self.tuned_model = PeftModel.from_pretrained(self.base_model, model_id).to(self.my_device)
        self.tuned_model = self.tuned_model.merge_and_unload()
        self.tuned_model.eval();

        # Create a pipeline
        self.pipe = pipeline("text-generation", model=self.tuned_model, tokenizer=self.base_tokenizer, max_new_tokens=max_length, temperature=temperature, top_p=top_p, top_k=top_k,
                         repetition_penalty=repetition_penalty, do_sample=do_sample, truncation=truncation, device=self.my_device)
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    
    def generate_response(self, prompt):
        """
        Generate text using the model with the provided prompt.
        """
        revised_prompt = self.__generate_instruction_chat_template(prompt)
        result = self.llm.invoke(revised_prompt)
        return result


    def __generate_instruction_chat_template(self, query_text):
        """
            Formats the query based on the instruction tuning prompt template.
        """

        # Apply the updated instruction-tuned prompt template
        formatted_prompt = f"""
        ### Instruction:
        You are an AI banking assistant. Respond to the customer's request in a clear and professional manner.

        ### Customer Request:
        {query_text}

        ### Response:
        """

        return formatted_prompt
    
    
if __name__ == "__main__":

    our_base_model_id = config.BASE_MODEL_ID
    our_pretrained_lm_path = config.MODEL_ID
    
    
    print(f"Our base model: {our_base_model_id}")
    print("-"*20)
    print(f"Our pre-trained model: {our_pretrained_lm_path}")
    print("-"*20)

    llm_wrapper = SmolLLMWrapper(our_base_model_id, our_pretrained_lm_path)

    # Test prompt
    prompt = "How can i get a new mobile password?"

    response = llm_wrapper.generate_response(prompt)
    print(f"Input Prompt: {prompt}\nResponse: {response}")