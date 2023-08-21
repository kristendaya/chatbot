import os
import sys
import tempfile

import openai
import streamlit as st
import pdfplumber
from io import BytesIO

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

st.title("PDF Chatbot")
uploaded_file = st.file_uploader(label="Upload a PDF file")

def process_pdf(uploaded_pdf):
    with pdfplumber.open(uploaded_pdf) as pdf:
        text = ' '.join(page.extract_text() for page in pdf.pages)
    return text

if uploaded_file:
    file_bytes = BytesIO(uploaded_file.getvalue())
    pdf_text = process_pdf(file_bytes)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
        tmpfile.write(pdf_text)
        tmpfile.close()
        loader = TextLoader(tmpfile.name)

        if PERSIST and os.path.exists("persist"):
            print("Reusing index...\n")
            vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            if PERSIST:
                index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
            else:
                index = VectorstoreIndexCreator().from_loaders([loader])

        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-3.5-turbo"),
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        )

        chat_history = []
        query = st.text_input("Enter your question:")

        if query:
            result = chain({"question": query, "chat_history": chat_history})
            st.write(result['answer'])

            chat_history.append((query, result['answer']))

        os.unlink(tmpfile.name)  # Delete the temporary file
