import streamlit as st
import os
import openai
from langchain_groq import ChatGroq
#from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory




from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
##os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Use the provided context to answer questions.\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])



#session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()


INDEX_PATH = "faiss_index"
PDF_FILE = "Customer.pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_vector_embedding(pdf_file=PDF_FILE, index_path=INDEX_PATH):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    
    #transformer
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)
    
    # Embedding 
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)

    # Save FAISS index for reuse
    st.session_state.vectors.save_local(index_path)



# Load or build FAISS ---->
def load_or_build_index(pdf_file=PDF_FILE, index_path=INDEX_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(index_path):
        st.session_state.vectors = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        create_vector_embedding(pdf_file, index_path)


# streamlit
st.title("Customer Support: Q&A")


load_or_build_index(PDF_FILE)

# retriever from db
retriever = st.session_state.vectors.as_retriever()


retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Rewrite the user question into a standalone query, "
     "using the conversation history if needed."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# Create history-aware retriever and doc chain

history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)

doc_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

# Wrap with message history
conversational_chain = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: st.session_state.chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


user_prompt=st.text_input("Enter your query from the PDF")

import time
if user_prompt:
    start = time.process_time()
    result = conversational_chain.invoke( {"input": user_prompt}, config={"configurable": {"session_id": "user1"}} )
    #st.write(f"⏱ Response time: {time.process_time()-start:.2f}s")
    st.markdown(f"**AI:**,  {result['answer']}")
    








