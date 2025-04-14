import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import torch # type: ignore
from torch import compile # type: ignore
from peft import PeftModel # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

import warnings
warnings.filterwarnings('ignore')

class SmolLLM_Raw_Wrapper:
    def __init__(self, base_model_id, model_id):
        
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
        self.tuned_model = torch.compile(self.tuned_model)
        self.tuned_model.eval();

    def generate_response(self, input_text, max_tokens=512, temp=0.3, top_p=0.9, top_k=50, penalty_score=1.2, do_sample=True):
        """
            Generates a response from the model based on the input.
        """

        inputs = self.__generate_instruction_chat_template(input_text)

        outputs = self.tuned_model.generate(
                    
            input_ids=inputs["input_ids"],  # Pass input_ids
            attention_mask=inputs["attention_mask"],  # Pass attention mask
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=penalty_score,
            do_sample=do_sample,
            pad_token_id=self.base_tokenizer.eos_token_id,
            eos_token_id=self.base_tokenizer.eos_token_id,
            early_stopping=True
        )

        # Decode output and clean it up
        output_text = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Ensure safe parsing without hardcoded token removal
        processed_output_text = output_text.strip()

        return processed_output_text
    
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

        #inputs = self.base_tokenizer.encode_plus(formatted_prompt, return_tensors="pt").to(self.my_device)
        inputs = self.base_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        inputs = {k: v.to(self.my_device) for k, v in inputs.items()}


        return inputs    
        
if __name__ == "__main__":

    our_base_model_id = config.BASE_MODEL_ID
    our_pretrained_lm_path = config.MODEL_ID
    
    
    print(f"Our base model: {our_base_model_id}")
    print("-"*20)
    print(f"Our pre-trained model: {our_pretrained_lm_path}")
    print("-"*20)

    llm_wrapper = SmolLLM_Raw_Wrapper(our_base_model_id, our_pretrained_lm_path)

    # Test prompt
    prompt = "How can i get a new mobile password?"

    # Generate response
    response = llm_wrapper.generate_response(prompt, max_tokens=512, temp=0.3, top_p=0.9, top_k=50, penalty_score=1.2, do_sample=True)

    print(f"Input Prompt: {prompt}\nResponse: {response}")