import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')

from langchain.chains import RetrievalQA # type: ignore
from langchain.prompts import PromptTemplate # type: ignore

class RAGChainBuilder:
    def __init__(self, llm, retriever):

        print("Initializing Retrieval Chain")
        print("-"*20)
        self.llm = llm
        self.retriever = retriever

        
        self.prompt_template = PromptTemplate.from_template(
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )

        self.prompt_template_v2 = PromptTemplate.from_template(
            """
            ### Instruction:
            You are an AI banking assistant. Respond to the customer's request in a clear and professional manner.

            ### Customer Request:
            {question}

            ### Context:
            {context}

            ### Response:
            """
        )

    def build_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt_template}
        )