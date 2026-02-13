# 👨‍🏫 Physics Mentor AI: RAG-Powered Educational Bot

Physics Mentor AI is a sophisticated **Retrieval-Augmented Generation (RAG)** chatbot specifically engineered to help Class 12 students master complex physics concepts. By integrating a local vector database with high-performance inference, the bot provides contextually accurate answers directly from specialized academic literature.

## 🚀 Key Features

* **Contextual Intelligence:** Unlike standard AI, this bot "reads" specific PDFs (Class 12 Physics) to provide factual, textbook-accurate answers.
* **Mentor Persona:** Custom-engineered prompts ensure a supportive, teacher-like tone using warm terms like "Beta" and "Dear Student."
* **High-Speed Inference:** Optimized using the **Groq Llama-3.3-70b-versatile** model for sub-second response times.
* **Vector Search:** Utilizes **FAISS** (Facebook AI Similarity Search) for efficient retrieval of complex scientific data.
* **Privacy-First API Management:** Designed to work with a modular `api.py` setup to keep sensitive keys secure.

## 🛠️ Tech Stack

* **Orchestration:** LangChain
* **Vector Database:** FAISS
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **LLM Provider:** Groq Cloud
* **Data Processing:** PyPDF & RecursiveCharacterTextSplitter


3. **Configure API Key:**
Create a file named `api.py` in the root directory:
```python
# api.py
api = "take groq free api key by login in the google account "

```


4. **Launch the Mentor:**
```bash
python main.py

```



## 🧠 How It Works

The system operates on a 5-step pipeline:

1. **Ingestion:** The `PyPDFLoader` extracts text from the textbook PDF.
2. **Chunking:** Text is broken into 1000-character segments with a 150-character overlap to preserve scientific context.
3. **Embedding:** HuggingFace models convert text chunks into high-dimensional numerical vectors.
4. **Retrieval:** When a student asks a question, FAISS identifies the top relevant chunks from the textbook.
5. **Augmentation:** The LLM receives the textbook context + the "Mentor" instructions to generate the final response.

## 👤 Author

**Sumit Soni** *First Year B.Tech Student (CS AI & ML) at KIET Group of Institutions*

