import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.title("⚖️ UAE Labor Law Assistant (RAG)")

# Set your API Key
api_key = st.text_input("Enter Gemini API Key", type="password")

if api_key:
    # 1. Load and Split PDF
    loader = PyPDFLoader("uae_labor_law.pdf")
    pages = loader.load_and_split()

    # 2. Setup Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(pages, embeddings)

    # 3. Setup QA Chain
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    user_query = st.text_input("Ask a question about your rights (e.g., Leave, Gratuity, Notice):")
    
    if user_query:
        response = qa.run(user_query)
        st.write(response)
