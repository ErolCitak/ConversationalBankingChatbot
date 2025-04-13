import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import config
from datasets import load_dataset # type: ignore
from langchain.docstore.document import Document # type: ignore

import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # Extract question and answer
        self.start_tag = "[INST]"
        self.end_tag = "[/INST]"

    def load_documents(self, split = 'train'):

        print(f"Initializing Dataset with {split} set")
        print("-"*20)
        raw_data = load_dataset(self.dataset_name)[split]
        data = []

        print("Dataset is preprocessing")
        print("-"*20)
        for text in raw_data:
            
            text = text['Prompts']
            start = text.find(self.start_tag) + len(self.start_tag)
            end = text.find(self.end_tag)

            question = text[start:end].strip()
            answer = text[end + len(self.end_tag):].replace("</s>", "").strip()

            # creating langchain document with Question and Answers
            document = Document(page_content=f"Question: {question}\nAnswer: {answer}")

            data.append(document)

        
        print("Dataset preprocessing is done")
        print("-"*20)

        return data    

if __name__ == "__main__":
    dataloader = DataLoader(config.DATASET_NAME)
    sample_docs = dataloader.load_documents(split='test')

    print("A sample document")
    print("-"*20)
    print(sample_docs[0])