import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_pdf_text(pdf_docs):  # get the text from the pdf
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):  # split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def create_vector_store(text_chunks):
    # Load the pre-trained model from Hugging Face using InstructorEmbedding
    model_name = 'hkunlp/instructor-xl'
    instructor = INSTRUCTOR(model_name)
    
    # Generate embeddings for the text chunks
    embeddings = []
    for chunk in text_chunks:
        embedding = instructor.encode(chunk)
        embeddings.append(embedding)
    
    # Create a vector store using FAISS
    embeddings = np.array(embeddings)
    vector_store = FAISS.from_embeddings(embeddings)
    return vector_store

def show_graphical_results():
    candidates = ["Anura Kumara", "Sajith Premadasa", "Ranil Wickremesinghe"]
    colors = ["red", "lightgreen", "darkgreen"]
    images = ["anura.jpg", "sajith.jpg", "ranil.jpg"]  # Replace with actual image paths
    results = [random.randint(0, 100) for _ in candidates]

    st.markdown("<style>" + open("styles.css").read() + "</style>", unsafe_allow_html=True)

    for candidate, color, image, result in zip(candidates, colors, images, results):
        st.markdown(f"""
        <div class="candidate-box">
            <div class="candidate-image" style="background-image: url('{image}');"></div>
            <div class="candidate-name">{candidate}</div>
            <div class="candidate-bar {color}" style="width: {result}%;"></div>
            <div class="candidate-result">{result}%</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with MACO", page_icon=":🗳️")

    st.header('Chat with MACO :🗳️')
    st.text_input('Ask anything about 2024 Election:')

    with st.sidebar:
        st.subheader("Statement of Policies")
        pdf_docs = st.file_uploader("Upload files regarding to 2024 Election and click Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get the PDF text
                raw_text = get_pdf_text(pdf_docs)
                
                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                # Create vector store
                vector_store = create_vector_store(text_chunks)
                st.success("Done!")

    if st.button("Show Graphical Results"):
            show_graphical_results()

if __name__ == "__main__":
    main()