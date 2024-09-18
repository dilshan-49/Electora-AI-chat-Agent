import os
import time
import shutil
import json
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
import tempfile
import plotly.express as px
from vertexai.generative_models import GenerativeModel
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.prompts import PromptTemplate
import streamlit as st

# Configurations and paths
CHROMA_PATH = './chroma'
GOOGLE_APPLICATION_CREDENTIALS = './vertexAIconfig.json'
PROJECT_ID = "electionchatbot-435710"
LOCATION = 'us-central1'
SAVE_PATH= './downloaded_files/'# Path to save the downloaded PDF
#fire store pdf directory
collection_name = 'pdfs'  # Firestore collection name

 

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

#spilt text to chunks
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

# Chat with the agent using the generate response function####################################################
def chat_with_agent():
    init_vertex_ai()
    st.set_page_config(page_title=" Chat with Agent ", page_icon="🤖", layout="wide")
    st.header("🤖 Chat with Agent 🤖")
    st.markdown("🚀 This is a tool to help you chat with an AI agent.🚀")

    # Load custom CSS
    with open("styles_Agent.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if "chat_history_agent" not in st.session_state:
        st.session_state.chat_history_agent = []
    
    # Create a container for chat messages
    chat_container = st.container()
    with chat_container:
        st.subheader("Chat with the Agent")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history_agent:
            st.markdown(f'<div class="chat-message user">{chat["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot">{chat["bot"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat with the agent
    user_input = st.text_input("You:", key="user_input")
    Prompt_Template_Agent = """This conversation is regarding to election in sri lanka and their candidates and their policies try to answer to user questions with everything you know.
    
    User: {user_input}
    """
    
    if user_input:
        # Run the agent
        prompt_agent = Prompt_Template_Agent.format(user_input=user_input)
        agent_response = generate_response(prompt_agent).text
        # Update chat history
        st.session_state.chat_history_agent.append({"user": user_input, "bot": agent_response})
        # Display the latest chat
        with chat_container:
            st.markdown(f'<div class="chat-message user">{user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot">{agent_response}</div>', unsafe_allow_html=True)
            
    if st.button("Back to Main Menu 🔙", key="Back to Main Menu"):
        st.session_state.current_function = "main"

# Main function to query and compare manifiestos###########################################################
def query_manifiesto():
    init_vertex_ai()
    st.set_page_config(page_title=" Query/Compare Manifiestos ", page_icon="📜", layout="wide")
    st.header("📜 Query/Compare Manifiestos 📜")
    st.markdown("🚀 This is a tool to help you query and compare the manifiestos of different political parties.🚀")
    st.markdown("📤 Upload the PDF files of the manifiestos and ask me any question about them. I will try to answer it!")
    st.write("📌 Note: Onece you upload the files,no need to upload again for a session.")
    # Load custom CSS
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

   

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
    if "chat_history_manifiesto" not in st.session_state:
        st.session_state.chat_history_manifiesto = []

    # Create a container for chat messages
    chat_container = st.container()

    # Display chat history
    with chat_container:
        st.subheader("Chat with the Bot")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history_manifiesto:
            st.markdown(f'<div class="chat-message user">{chat["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot">{chat["bot"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    query = st.text_input("Enter your question here:")
    if st.button("Back to Main Menu 🔙", key="Back to Main Menu"):
            st.session_state.current_function = "main"
    
    if query:
        results = search_chroma(query)
                
        if not results or results[0][1] < 0.5:
            st.write("Sorry, I don't know the answer to that question.")
            return
        
        chat_history_manifiesto = '\n'.join([f"User:{chat['user']}\nBot: {chat['bot']}"for chat in st.session_state.chat_history_manifiesto])
        
        context_text = '\n\n---\n\n'.join([doc.page_content for doc, _ in results])
                
        prompt_template = """
                Use the following pieces of context and chat history to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                {context}
                Chat History:
                {chat_history}

                Question: {question}
                """
                
        prompt = prompt_template.format(context=context_text,chat_history=chat_history_manifiesto, question=query)
        st.write("Generating response from Vertex AI...")
        response = generate_response(prompt=prompt)
        answer = response.text
                
                # Update chat history
        st.session_state.chat_history_manifiesto.append({"user": query, "bot": answer})

                # Display the latest chat
        with chat_container:
            st.markdown(f'<div class="chat-message user">{query}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot">{answer}</div>', unsafe_allow_html=True)
            st.write(results[0][1])

# functionS for predict the winner###################################################################

# Analyze the chunk using your model and return the results
def analyze_chunk(chunk, model):
    # Structured prompt for model to generate predictions
    prompt = f"Output should be dictionary data type and give only the dictionary.dont try to give enything else and use leader's full name in dictionary. Analyze the following text and predict support percentages for each leader in the format {{'Leader Name': percentage}}:\n\n{chunk}"
    prompt2=f" Analyze the following text and predict what are the trends in support for each leader and who is likely to win in presidential Election. You may use your information sources too. but prioritize these data :\n\n{chunk}"
    results = model.generate_content([prompt]).text
    results2= model.generate_content([prompt2]).text
    # Debug: Print the raw result to see what the model is returning
    #st.write("Raw result:", results)

    # Check if the result is empty
    if not results.strip():
        st.write("Model returned an empty response.")
        return {}

    # Clean up the result if needed
    results = results.strip()  # Remove leading/trailing spaces
    results = results.replace("```python", "").replace("```", "").replace("'", '"') # Replace single quotes with double quotes for valid JSON

    # Debug: Print cleaned result
    #st.write("Cleaned result:", results)

    try:
        # Attempt to parse the result as JSON
        results_dict = json.loads(results)
    except json.JSONDecodeError as e:
        # If there's a parsing error, print the error and return a default value
        st.write(f"Error parsing the model output: {e}")
        results_dict = {}

    # Display the final dictionary for verification
    #st.write("Parsed dictionary:", results_dict)
    return results_dict,results2

 
#get Average resoults for each party
def avg_results(results):
    party_scores = {}
    for result in results:
        for party, score in result.items():
            if party not in party_scores:
                party_scores[party] = 0
            party_scores[party] += score

    total_score = sum(party_scores.values())
    party_percentages = {party: (score / total_score) * 100 for party, score in party_scores.items()}
    return party_percentages

#win predictor
def win_predict():
    init_vertex_ai()
    st.set_page_config(page_title='Win_Predict',page_icon='🏆',layout='wide')
    st.header('🏆 Predict Win 🏆')
    st.markdown('🚀 This is a tool to help you predict the winning party based on the current situation.🚀')
    
    #return button
    if st.button("Back to Main Menu 🔙", key="Back to Main Menu"):
        st.session_state.current_function = "main"
    
    survey_pdf = st.file_uploader("Choose survey files", type="pdf", accept_multiple_files=True)
    if st.button('📈 Predict',key = 'predict'):
        if survey_pdf:
            with st.spinner("Processing...."):
                surveys = load_pdf(survey_pdf)

                predict_model = GenerativeModel("gemini-1.5-flash-001")
                survay_results=[]

                #for survey in surveys:
                survay_result,prediction=analyze_chunk(surveys,predict_model)
                survay_results.append(survay_result) 
                st.write(survay_result)               
                    #time.sleep(3)
                
                final_percentages =avg_results(survay_results)
                st.success("Analysis Complete! ")

                st.subheader("Winning Percentages for Each Party")

                # Display the percentages as text
                for party, percentage in final_percentages.items():
                    st.write(f"{party}: {percentage:.2f}%")

                # Create a bar chart
                fig = px.bar(
                    x=list(final_percentages.keys()),
                    y=list(final_percentages.values()),
                    labels={'x': 'Party', 'y': 'Percentage'},
                    title="Winning Percentages for Each Party"
                )

                # Display the bar chart
                st.plotly_chart(fig)
                with st.container():
                    st.write(prediction)


def main():
    init_vertex_ai()
    st.set_page_config(page_title="Election ChatBot", page_icon="🗳️", layout="wide")
    st.header("Election ChatBot: 🗳️")
    st.markdown("ASK me anything about the election and I will try to answer it!")
    
     # Create columns for the buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📜 Query/Compare Manifiestos 📜",key ="Query Manifiesto"):
            st.session_state.current_function  = "query_manifiesto"
        
    with col2:
        if st.button("🤖 Chat with Agent 🤖",key = "Chat with Agent"):
            st.session_state.current_function = "chat_with_agent"
            
    with col3:
        if st.button("🏆 Predict Win 🏆",key = "Predict Win"):
            st.session_state.current_function = "win predict"

    st.header("Wellcome to Election ChatBot. We are AI powered bots to help you with the election queries.")
    st.write("Please follow the below steps to use the Election ChatBot ⬇️")
    st.write("         1. Click on the Query/Compare Manifiestos button 📜 to upload the manifiestos of different political parties and ask questions about them.")
    st.write("         2. Click on the Chat with Agent button 🤖 to chat with the AI agent about election and candidate's details.")
    st.write("         3. Click on the Predict Win button 🏆 to predict the winning party based on current situation.")
# Execute the main function
if __name__ == "__main__":
    if "current_function" not in st.session_state:
        st.session_state.current_function = 'main'
    if st.session_state.current_function == "main":
        main()
    elif st.session_state.current_function == "query_manifiesto":
        query_manifiesto()
    elif st.session_state.current_function == "chat_with_agent":
        chat_with_agent()
    elif st.session_state.current_function == "win predict":
        win_predict()