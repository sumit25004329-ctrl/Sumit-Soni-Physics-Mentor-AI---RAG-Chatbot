import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
import api

# PDF read karke raw text data load kar raha hai
loader = PyPDFLoader("Class/leph101.pdf")
data = loader.load()

# Text ko small chunks me split karta hai taaki relevance bani rahe
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# Text ko numerical vectors me convert karne ke liye HuggingFace model ka use
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vectors ko locally RAM me store karta hai fast retrieval ke liye
vectorstore = FAISS.from_documents(docs, embeddings)

# AI ko specific character (Physics Mentor) dene ke liye prompt template
mentor_template = """
You are a friendly and wise Physics Mentor. 
The person talking to you is your Student.

IDENTITY RULES:
- If the student asks "Who are you?" or "What is your name?", you must answer: "I am your Physics Mentor, created to help you master Class 12 Physics!"
- If the student asks "Who created you?", mention: "I was developed by Sumit Soni."

INSTRUCTIONS:
1. Use the context below to answer physics questions.
2. Be encouraging and use terms like 'Beta' or 'Dear Student'.
3. Always end with a follow-up question to check the student's understanding.

Context: {context}
Student's Question: {question}

Mentor's Response:"""

PROMPT = PromptTemplate(
    template=mentor_template, 
    input_variables=["context", "question"]
)

# Here I use Groq freeApi and store it in other file and import its api file with keyword
llm = ChatGroq(
    groq_api_key=api.api, # API key yahan jayegi
    model_name="llama-3.3-70b-versatile"
)

# Query aur PDF content ko connect karne wali main chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat Loop until Student write 'sumitsoni'
print("\n" + "="*50)
print("Physics Mentor by SUMIT SONI ")
print("Write 'sumitsoni' for exit from coversation.")
print("="*50 + "\n")

while True:
    user_input = input("Student: ")

    if user_input.lower()=='sumitsoni':
        print("\nMentor: Keep practicing, beta! Physics is the key to understanding the universe. Goodbye!")
        break

    try:
        response = qa_chain.invoke({"query": user_input})
        print(f"\nMentor: {response['result']}")
        print("\n" + "-"*30 + "\n")
    except Exception as e:
        print(f"\nMentor: Sorry beta, I can't reply : {e}")


