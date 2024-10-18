import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import numpy as np
import os

# Import the updated Google Generative AI integration
from langchain_google_genai import ChatGoogleGenerativeAI


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name, trust_remote_code=False):
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode(text)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store_from_multiple_pdfs(pdf_docs):
    all_text_chunks = []
    embeddings = SentenceTransformerEmbeddings("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
    for pdf in pdf_docs:
        pdf_text = get_pdf_text([pdf])
        text_chunks = get_text_chunks(pdf_text)
        all_text_chunks.extend(text_chunks)

    vector_store = FAISS.from_texts(all_text_chunks, embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key='',
                                 temperature=0.2)

    template = """Answer the question based on the following context:
    {context}

    Question: {question}

    If the answer is not in the context, just say "Let me gather more information so I can provide you with the best answer."

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    retriever = vector_store.as_retriever()

    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


def user_input(user_question):
    response = st.session_state.conversation.invoke(user_question)
    st.write("Human: ", user_question)
    st.write("Bot: ", response)


def main():
    st.set_page_config('chat with multiple pdf')
    st.header('RandomTrees Chatbot 💬')
    user_question = st.text_input("Ask a Question")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title('settings')
        st.subheader('Upload your documents')
        pdf_docs = st.file_uploader('upload your PDF files:', accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store_from_multiple_pdfs(pdf_docs)
                st.session_state.conversation = get_conversation_chain(vector_store)
                st.success("Done")


if __name__ == "__main__":
    main()

