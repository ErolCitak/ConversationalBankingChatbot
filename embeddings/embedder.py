import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.docstore.document import Document # type: ignore

import warnings
warnings.filterwarnings('ignore')

class Embedder:
    def __init__(self, model_name):
        print("Initializing Embedding Model")
        print("-"*20)
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_text(self, text=""):
        embedding_vector = self.embedding_model.embed_query(text) # return List[float]

        return embedding_vector

    def build_vectorstore(self, documents):
        return FAISS.from_documents(documents, self.embedding_model)
    

if __name__ == "__main__":

    ## Embedder model name is taken from config.py
    embedder_name = config.EMBED_MODEL

    # Embedder class is initializing
    embedder = Embedder(embedder_name)

    # Sample embedding of a text
    sample_text = "This is a sample text to verify embed query works well."
    sample_embedding_vector = embedder.embed_text(sample_text)

    print(f"Shape of the embedding: {len(sample_embedding_vector)}")
    print("-"*20)

    # Sample embedding of a document
    sample_data = []

    for i in range(10):
        sample_data.append(Document(page_content=f"Question: Sample Question_{i}\nAnswer: Sample_Answer_{i}"))
    
    sample_vector_store = embedder.build_vectorstore(sample_data)
    print(f"There are {sample_vector_store.index.ntotal} documents in the vector store")
    print("-"*20)

    # Let's make a dummy search over the vector store
    dummy_search_text = "7_AnSwEr"
    results = sample_vector_store.similarity_search(dummy_search_text, k=1)
    print(f"Similarity Search Over Vector Store\nSearch Text: {dummy_search_text}\nMost Relevant Return: {results} ")
    print("-"*20)