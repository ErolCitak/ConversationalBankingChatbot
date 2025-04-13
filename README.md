# ConversationalBankingChatbot  
  
  
ConversationalBankingChatbot/  
│  
├── main.ipynb                   # Ana giriş noktası (Chat/QA başlatıcı)  
├── config.py                    # Model, dosya yolu, ayarlar  
├── logging_utils.py             # Loglama, zaman tutucu, helper fonksiyonlar  
│  
├── data/  
│   └── prepare_data.py          # Dataset yükleme ve LangChain'e uygun hâle getirme  
│  
├── embeddings/  
│   └── embedder.py              # Embedding modeli ve vector store hazırlama  
│  
├── llm/  
│   └── smollm_wrapper.py        # SmolLM2 modelini LangChain LLM'e saran sınıf  
│  
├── chains/  
│   └── rag_chain.py             # RetrievalQA zinciri burada kuruluyor  
│  
├── interface/  
│   └── gradio_ui.py             # Gradio, Streamlit ya da FastAPI arayüzü  