import os
import shutil
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
import tempfile
from vertexai.generative_models import GenerativeModel
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import PromptTemplate
import streamlit as st# Configurations and paths

CHROMA_PATH = './chroma'
GOOGLE_APPLICATION_CREDENTIALS = './vertexAIconfig.json'
PROJECT_ID = "electionchatbot-435710"
LOCATION = 'us-central1'

# Initialize Vertex AI
def init_vertex_ai():
    credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
    aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
    vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load and split PDF documents
def load_pdf(uploaded_files):
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(temp_file_path)
            documents.extend(loader.load())
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Save chunks to Chroma vector database
def save_to_chroma(chunks):
    # Clean up old database
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError as e:
            st.error(f"Error cleaning up old database: {e}")
            return


    vertex_embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    
    # Create and persist Chroma database
    db = Chroma.from_documents(
        chunks, vertex_embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    return db

# Search in ChromaDB
def search_chroma(query, k=4):
    vertex_embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=vertex_embeddings)
    return db.similarity_search_with_score(query, k=k)

# Generate response from Vertex AI
def generate_response(prompt):
    model = GenerativeModel("gemini-1.5-flash-001")
    response = model.generate_content([prompt])
    return response

# Main function
def main():
    st.set_page_config(page_title="Election ChatBot", page_icon="🗳️", layout="wide")

    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    col1, col2 = st.columns([15, 6])
    with col1:
        st.header("Election ChatBot: 🗳️")
    with col2:
        st.button("Chat with Agent")
    st.markdown("ASk me anything about the election and I will try to answer it!")

    with st.sidebar:
        st.subheader("Upload PDF Files: 📄")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    init_vertex_ai()
                    
                    st.write("Loading and splitting PDF documents...")
                    documents = load_pdf(uploaded_files)
                    chunks = split_text(documents)
                    
                    st.write(f"Saving {len(chunks)} chunks to ChromaDB...")
                    save_to_chroma(chunks)
                    
                    st.success("Processing complete!")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Create a container for chat messages
    chat_container = st.container()

    # Display chat history
    with chat_container:
        st.subheader("Chat with the Bot")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="chat-message user">{chat["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot">{chat["bot"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    query = st.text_input("Enter your question here:")
    
    if query:
        results = search_chroma(query)
                
        if not results or results[0][1] < 0.5:
            st.write("Sorry, I don't know the answer to that question.")
            return
        
        chat_history = '\n'.join([f"User:{chat['user']}\nBot: {chat['bot']}"for chat in st.session_state.chat_history])
        context_text = '\n\n---\n\n'.join([doc.page_content for doc, _ in results])
                
        prompt_template = """
                Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                {context}
                Chat History:
                {chat_history}

                Question: {question}
                """
                
        prompt = prompt_template.format(context=context_text,chat_history=chat_history, question=query)
        st.write("Generating response from Vertex AI...")
        response = generate_response(prompt=prompt)
        answer = response.text
                
                # Update chat history
        st.session_state.chat_history.append({"user": query, "bot": answer})

                # Display the latest chat
        with chat_container:
            st.markdown(f'<div class="chat-message user">{query}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot">{answer}</div>', unsafe_allow_html=True)
            st.write(results[0][1])

# Execute the main function
if __name__ == "__main__":
    main()
    
