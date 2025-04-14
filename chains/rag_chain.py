import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')

from langchain.chains import RetrievalQA # type: ignore

class RAGChainBuilder:
    def __init__(self, llm, retriever):
        
        print("Initializing Retrieval Chain")
        print("-"*20)
        self.llm = llm
        self.retriever = retriever

    def build_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff"
        )