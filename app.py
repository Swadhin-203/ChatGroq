import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit session state setup
if "vector" not in st.session_state:
    st.session_state.embeddings = embeddings
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    
    # Store in session_state as "vector" not "vectors"
    st.session_state.vector = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# App UI
st.title("ChatGroq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")



# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

# Retrieval and chaining
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input prompt
prompt_input = st.text_input("Input your prompt here")

if prompt_input:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt_input})
    st.write("⏱️ Response time: ", round(time.process_time() - start, 2), "seconds")
    st.write(response['answer'])

    # Optional: display relevant documents
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------")
