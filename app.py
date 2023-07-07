from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader , Docx2txtLoader, CSVLoader
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from langchain.chains import ConversationalRetrievalChain

# Set your OpenAI API key
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Use the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


st.set_page_config(page_icon=":robot:", page_title="Q&A Chatbot using RAG" , layout="wide")
st.title("Q&A Chatbot using RAG 🚀")
st.subheader("Upload your text source file and ask questions about its content.")

uploaded_file = st.file_uploader("Choose your source file", type=["pdf" , "txt" , "docx" , "csv"])


documents = []
if uploaded_file is not None:
    # Read the PDF file
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file)
        documents.extend(loader.load())
    elif uploaded_file.endswith('.docx') or uploaded_file.endswith('.doc'):
        loader = Docx2txtLoader(uploaded_file)
        documents.extend(loader.load())
    elif uploaded_file.endswith('.txt'):
        loader = TextLoader(uploaded_file)
        documents.extend(loader.load())

    # Extract text from the PDF
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)


    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)

    # Download embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create the document search
    vectordb = FAISS.from_texts(texts, embeddings)

    vectordb.persist()
    
    # Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo"),
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        verbose=False,
    )

    # Initialize chat history list
    chat_history = []

    # Custom set of questions 
    
    # Get the user's query
    query = st.text_input("Enter your question about the uploaded document:")

    # Add a generate button
    generate_button = st.button("Generate Answer")

    if generate_button and query:
        with st.spinner("Generating answer..."):
            result = qa({"question": query, "chat_history": chat_history})

            answer = result["answer"]
            source_documents = result['source_documents']

            # Combine the answer and source_documents into a single response
            response = {
                "answer": answer,
                "source_documents": source_documents
                        }
            st.write("response:", response)