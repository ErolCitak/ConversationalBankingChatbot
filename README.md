# ConversationalBankingChatbot  
  
  
ConversationalBankingChatbot/  
│  
├── main.ipynb                   # Main Entrance  
├── config.py                    # Some constants definitions about the LLM Model, Dataset, Embedder, etc.  
├── logging_utils.py             # Logging Utilities  
│  
├── data/  
│   └── prepare_data.py          # Download, load, and making ready for LangChain Document  
│  
├── embeddings/  
│   └── embedder.py              # Initializing embedder and creation of VectorStore  
│  
├── llm/  
│   └── smollm_wrapper.py        # LangChain LLM of previously pre-trained SmolLM2 (our model)  
│  
├── chains/  
│   └── rag_chain.py             # Chain of Retrieval Augmented Generation initialization  
│  
├── interface/  
│   └── gradio_ui.py             # Gradio, Streamlit or FastAPI interface  