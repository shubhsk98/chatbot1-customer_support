import streamlit as st
import os
from langchain_groq import ChatGroq
#from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

#session state
if "vectors" not in st.session_state:
        st.session_state.vectors = None


def create_vector_embedding(pdf_file="Customer.pdf", index_path="faiss_index"):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    
    #transformer
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)
    
    # Embedding 
    embeddings = HuggingFaceEmbeddings()
    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)

    # Save FAISS index for reuse
    st.session_state.vectors.save_local(index_path)



# Load or build FAISS ---->
def load_or_build_index(pdf_file="Customer.pdf", index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings()

    if os.path.exists(index_path):
        st.session_state.vectors = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        create_vector_embedding(pdf_file, index_path)


st.title("Customer Support: Q&A")

load_or_build_index("Customer.pdf")

user_prompt=st.text_input("Enter your query from the PDF")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])






