from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader , Docx2txtLoader, CSVLoader
from langchain.document_loaders import PyPDFLoader
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
import pytesseract
from pdf2image import convert_from_path
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


st.sidebar.title("About")
st.sidebar.info("This is a custom knowledge base QnA Chatbot which leverages the power of RAG (Retrieval-Augmented Generation) module to generate answers to your questions.")



uploaded_file = st.file_uploader("Choose your source file", type=["pdf" , "txt"])

def pdf_to_text(pdf_path):
    # Step 1: Convert PDF to images
    images = convert_from_path(pdf_path)

    with open('output.txt', 'w') as f:  # Open the text file in write mode
        for i, image in enumerate(images):
            # Save pages as images in the pdf
            image_file = f'page{i}.jpg'
            image.save(image_file, 'JPEG')

            # Step 2: Use OCR to extract text from images
            text = pytesseract.image_to_string(image_file)

            f.write(text + '\n')  # Write the text to the file and add a newline for each page


def load_txt_data(uploaded_file):
    with open('uploaded_file.txt', 'w') as f:
        f.write(uploaded_file.getvalue().decode())
    return uploaded_file.getvalue().decode()


# Splitter  to Split the text into smaller chunks
text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

documents = []
if uploaded_file is not None:
    # Read the PDF file as text
    if uploaded_file.type == "text/plain":
        doc = "text"
        data = load_txt_data(uploaded_file)
        loader = TextLoader('uploaded_file.txt')
        documents.extend(loader.load())
        texts = text_splitter.split_documents(documents)

    elif uploaded_file.type == "application/pdf":
        doc_reader = PdfReader(uploaded_file)
        # Extract text from the PDF
        raw_text = ""
        for i, page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        texts = text_splitter.split_text(raw_text)

    # Download embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create the document search
    vectordb = FAISS.from_texts(texts, embeddings)
    
    # Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(streaming=True , temperature=0.2, openai_api_key=OPENAI_API_KEY),
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
            st.write("Response Body:", response)
            st.write("Answer:", response["answer"])