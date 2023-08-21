import streamlit as st
import pinecone
import pdfplumber
import openai
from io import BytesIO

# Pinecone and OpenAI API keys
pinecone.init(api_key="")
openai.api_key = ""

# Pinecone namespace
pinecone.deinit()
pinecone.init(api_key="")
NAMESPACE = "madebygpt"
pinecone.create_namespace(NAMESPACE)


def process_pdf(uploaded_pdf):
    with pdfplumber.open(uploaded_pdf) as pdf:
        return ' '.join(page.extract_text() for page in pdf.pages)


def upload_pdf_to_pinecone(pdf_text, pdf_name):
    vector = get_vector_using_GPT-3_request_function(pdf_text)
    pinecone.upsert(upserts={pdf_name: vector}, namespace=NAMESPACE)


def search_pinecone(query_text):
    vector = get_vector_using_GPT-3_request_function(query_text)
    results, ids = pinecone.fetch(ids=None, query_vector=vector, top_k=1, namespace=NAMESPACE)

    # Get the id with the highest score
    pdf_name = ids[0][0]

    return pdf_name


st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF")

if uploaded_file:
    file_bytes = BytesIO(uploaded_file.getvalue())
    pdf_text = process_pdf(file_bytes)

    pdf_name = uploaded_file.name
    upload_pdf_to_pinecone(pdf_text, pdf_name)
    st.success(f"PDF uploaded: {pdf_name}")

query = st.text_input("Enter your question:")

if query:

    pdf_result_name = search_pinecone(query)
    st.write(f"The most relevant PDF is {pdf_result_name}")

pinecone.deinit()
pinecone.delete_namespace(NAMESPACE)
