import os
import json
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
import tempfile
import plotly.express as px
from vertexai.generative_models import GenerativeModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
import streamlit as st
import firebase as fb
import vote as vt

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
    vertex_embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003")
    
    # Check if the Chroma database exists
    if os.path.exists(CHROMA_PATH):
        # Load the existing database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=vertex_embeddings)
        # Add new documents to the existing database
        db.add_documents(chunks)
    else:
        # Create a new database if it doesn't exist
        db = Chroma.from_documents(
            chunks, vertex_embeddings, persist_directory=CHROMA_PATH
        )
    
    # Persist the database
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

# Chat with the agent using the generate response function
def chat_with_agent():
    
    init_vertex_ai()
    
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
   
    chat_history_agent = '\n'.join([f"User:{chat['user']}\nBot: {chat['bot']}"for chat in st.session_state.chat_history_agent])
    Prompt_Template_Agent = """This conversation is regarding to political based elections in sri lanka and their parties and leaders try to answer to user questions using  everything you know.
    if user questions doesnt match with articles try to answer with yourself,chat history:{chat_history} use this chat history to memorize the conversation and answer to user questions.
    User: {user_input}
    
   

    """
    
    if user_input:
        

        # Run the agent model to generate a response
        prompt_agent = Prompt_Template_Agent.format(user_input=user_input, 
                                                    chat_history = chat_history_agent,
                                                    
                                                    )
        
        agent_response = generate_response(prompt_agent).text
        # Update chat history
        st.session_state.chat_history_agent.append({"user": user_input, "bot": agent_response}  )
        # Display the latest chat
        with chat_container:
            st.markdown(f'<div class="chat-message user">{user_input}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-message bot">{agent_response}</div>', unsafe_allow_html=True)

         
    
# Main function to query and compare manifiestos###########################################################
def query_manifiesto():
    
    init_vertex_ai()
    st.set_page_config(page_title=" Query/Compare Manifiestos ", page_icon="📜", layout="wide")
   
    st.markdown(
        """
        <style>
        .white-space-bg {
            background-image: url("https://easemybusiness.com/wp-content/uploads/2020/06/Political-canvassing-banner.jpg");
            background-size: cover;
            background-position: center;
            border-radius: 10px;
            padding: 40px;
            color: white;
            font-size: 18px;
        }
        .blurred-box {
        background: rgba(255, 255, 255, 0.3); /* White background with transparency */
        border-radius: 10px; /* Rounded corners */
        padding: 20px; /* Padding inside the box */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        backdrop-filter: blur(10px); /* Blur effect */
        -webkit-backdrop-filter: blur(10px); /* Blur effect for Safari */
        margin: 20px auto; /* Center the box */
        max-width: 600px; /* Maximum width of the box */
    }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<h1 style='text-align: center; color: #FFA500;'>📜Query and Compare the manifiesto </h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class='white-space-bg'>
            <div class='blurred-box'>
               <ol style='margin: 0 auto; max-width: 600px; color: black;'>
                    <li><b>🚀 This is a tool to help you query and compare the manifestos of different political parties.🚀.</b></li>
                    <li><b>📤 Upload the PDF files of the manifestos using sidebar and ask me any question about them. I will try to answer it!.</b></li>
                    <li><b>📌 Note: Once you upload the files, no need to upload again for a session.</b></li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True )
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
        chunks = split_text(load_pdf(uploaded_files))        
        if not results or results[0][1] < 0.5:
            st.write("Sorry, I don't know the answer to that question.")
            return
        
        chat_history_manifiesto = '\n'.join([f"User:{chat['user']}\nBot: {chat['bot']}"for chat in st.session_state.chat_history_manifiesto])
        
        context_text = '\n\n---\n\n'.join([doc.page_content for doc, _ in results])
                
        prompt_template = """
                Use the following pieces of context and chat history to answer the question at the end. If you don't know the answer try to answer using {docs_vectors}, just say that you don't know, don't try to make up an answer.

                {context}
                Chat History:
                {chat_history}

                Question: {question}
                """
                
        prompt = prompt_template.format(context=context_text,chat_history=chat_history_manifiesto, question=query,docs_vectors=chunks)
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
    st.markdown("<h2 style='text-align: center; color: #FFA500;'> 🏆 Predict Win 🏆</h2>",
                unsafe_allow_html=True
    )
    st.markdown('🚀 This is a tool to help you predict the winning party based on the current situation.🚀')
    
    #return button
    if st.button("Back to Main Menu 🔙", key="Back to Main Menu"):
        st.session_state.current_function = "main"
    
    #vote for party
    st.markdown('🗳️ Vote for your favorite party 🗳️')
    party1,party2,party3,party4 = st.columns(4)
    #add three buttons
    with party1:
        st.markdown(
            """
            <div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 10px;'>
                <h3 style='text-align: center;'>
                    <img src='https://srilankabrief.org/wp-content/uploads/2014/03/428447719unp5.jpg' alt='NPP Logo' width='50' height='50' style='border-radius: 5px;'>
                    United National Party
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button('Vote for UNP', key='UNP'):
            vt.increment_vote_count('UNP')
            st.write('You voted for United National Party')
    with party2:
        st.markdown(
            """
            <div style='border: 2px solid #FF9800; padding: 10px; border-radius: 10px;'>
                <h3 style='text-align: center;'>
                    <img src='https://th.bing.com/th/id/OIP.XkIWWZc2j8NH-_g7hIhUhQHaE7?w=248&h=180&c=7&r=0&o=5&dpr=1.1&pid=1.7' alt='NPP Logo' width='50' height='50' style='border-radius: 5px;'>
                    Sri Lanka Podujana Peramuna
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button('Vote for SLPP', key='SLPP'):
            vt.increment_vote_count('SLPP')
            st.write('You voted for Sri Lanka Podujana Peramuna')
    with party3:
        st.markdown(
            """
            <div style='border: 2px solid #FFEB3B; padding: 10px; border-radius: 10px;'>
                <h3 style='text-align: center;'>
                    <img src='https://assets.manthri.lk/uploads/party/logo/9/medium_SJB-f07ef5f8f1b469e3637d173503536657.jpg' alt='NPP Logo' width='50' height='50' style='border-radius: 5px;'>
                    Samagi Jana Balawegaya
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button('Vote for SJB', key='SJB'):
            vt.increment_vote_count('SJB')
            st.write('You voted for Samagi Jana Balawegaya')
    with party4:
        st.markdown(
            """
            <div style='border: 2px solid #F44336; padding: 10px; border-radius: 10px;'>
            <h3 style='text-align: center; border-radius: 10px;'>
                <img src='https://th.bing.com/th/id/OIP.gGO_SJaqX0UXVuj7EOG-8AAAAA?w=182&h=181&c=7&r=0&o=5&dpr=1.1&pid=1.7' alt='NPP Logo' width='50' height='50' style='border-radius: 5px;'>
                National People’s Power
            </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button('Vote for NPP', key='NPP'):
            vt.increment_vote_count('NPP')
            st.write('You voted for National People’s Power')
    #get the votes
    if st.button('Analys Voting',key = 'get_votes'):
        votes = vt.get_vote_counts()
        #plot using plotly
        fig = px.pie(
            values=list(votes.values()),
            names=list(votes.keys()),
            title='Votes for Each Party'
        )
        st.plotly_chart(fig)
        analysis = f'analys the votes and predict the winning party based on the given data{votes}'
        st.write(generate_response(analysis).text)
    survey_pdf = st.file_uploader("Choose survey files", type="pdf", accept_multiple_files=True)
    if st.button('📈 Predict',key = 'predict'):
        if survey_pdf:
            with st.spinner("Processing...."):
                surveys = load_pdf(survey_pdf)

                predict_model = GenerativeModel("gemini-1.5-flash-001")
                survay_results=[]

                #for survey in surveys:
                survay_result,prediction =analyze_chunk(surveys,predict_model)
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
def electionNews():
    import newsQueryfunc as nf
    
    db = fb.db
    st.set_page_config(page_title='Election News', page_icon='📰', layout='wide')
    st.markdown("<h1 style='text-align: center;'> 📰 Election News 📰 </h1>", unsafe_allow_html=True)


    # Back to main menu button
    col0,col1, col2 = st.columns([2, 4, 1])  # Adjust column sizes to push button to the right
    with col1:
            st.image("https://img.freepik.com/premium-vector/news-tiny-people-read-breaking-news-newspaper-modern-flat-cartoon-style-vector-illustration_501813-452.jpg")
    with col2:
        if st.button("Back to Main Menu 🔙"):
            st.session_state.current_function = "main"



    if 'latest_news' not in st.session_state:
        articles = nf.fetch_latest_articles(db)

        try:
            with st.spinner("Summarizing news..."):
                summaries = "\n".join([f"{article['title']}: {article['content']}" for article in articles])
                prompt = (
                    f'''Using the following recent election articles and your knowledge base on recent Sri Lankan political news RELATED election, to provide summary on latest news. 
                    If you do not have enough information, kindly state that you lack sufficient information. 
                    \n\nYour response should be structured as a well-organized description, with clear points and paragraphs. 
                    Please format your output in HTML, including appropriate headings, paragraphs, and lists, so it can be displayed on a web page.
                    \n\nRecent Articles: {summaries}'''
                )
                response=generate_response(prompt)
                st.markdown(response.text, unsafe_allow_html=True)
                st.session_state.latest_news = response.text
        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        # Display the latest news
        st.markdown(st.session_state.latest_news, unsafe_allow_html=True)
        
    user_query = st.text_input("Enter your question or keyword:", key="user_query")

    if st.button("Submit", key="submit_query"):
        if user_query:
            # If user submits a query, fetch relevant news instead of latest news
            with st.spinner("Fetching relevant articles..."):
                prompt=f"Generate list of keywords to search for relevant news articles based on the user query.\n\nOutput must be a python list with string elements.\n\nUser Query: {user_query}"

                articles = nf.get_relevant_news(user_query,db)
                if not articles:
                    with st.spinner("Searching for news..."):
                        prompt = f'''No relevant articles found in the database for the question.Answer based on your knowledge on recent Sri Lankan political newse. 
                            \n\nIf you dont have related information, politely say you don't have information .
                            \n\nYour response should be structured as a well-organized description, with clear points and paragraphs to address the question. 
                            Please format your output in HTML, including appropriate headings, paragraphs, and lists, so it can be displayed on a web page.
                            \n\nQuestion: {user_query}
                            '''
                        response=generate_response(prompt)
                        
                        st.markdown(response.text, unsafe_allow_html=True)

                else:
                    with st.spinner("Summarizing news..."):
                        summaries = "\n\n".join([f"{article['title']}: {article['content']} \n {article['url']}" for article in articles])
                        prompt = (
                        f'''Using the following recent election articles and your knowledge base on recent Sri Lankan political news, please answer the question below. 
                        If you do not have enough information, kindly state that you lack sufficient information. 
                        \n\nYour response should be structured as a well-organized description, with clear points and paragraphs to address the question. 
                        Please format your output in HTML, including appropriate headings, paragraphs, and lists, so it can be displayed on a web page.
                        \nIn the of response provide url's of most relevant articles for reference.
                        \n\nRecent Articles: {summaries} 
                        \n\nQuestion: {user_query}'''
                        )
                        response=generate_response(prompt)
                        st.markdown(response.text, unsafe_allow_html=True)

def main():
    # Set up the page configuration
    st.set_page_config(
        page_title="ELECTORA ",
        page_icon="🗳️",
        layout="wide"
    )
    
    # Apply custom CSS styling to add a background image to the white space area
    st.markdown(
        """
        <style>
        .white-space-bg {
            background-image: url("https://t4.ftcdn.net/jpg/05/57/43/29/360_F_557432985_C16r8R1CNl4YBAazkvhRR3r9f4IaLZdm.jpg");
            background-size: cover;
            background-position: center;
            border-radius: 10px;
            padding: 40px;
            color: white;
            font-size: 18px;
        }
        .blurred-box {
        background: rgba(255, 255, 255, 0.3); /* White background with transparency */
        border-radius: 10px; /* Rounded corners */
        padding: 20px; /* Padding inside the box */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        backdrop-filter: blur(10px); /* Blur effect */
        -webkit-backdrop-filter: blur(10px); /* Blur effect for Safari */
        margin: 20px auto; /* Center the box */
        max-width: 600px; /* Maximum width of the box */
    }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header section with a larger font size and center alignment
    st.markdown(
        "<h1 style='text-align: center; color: #FFA500;'>ELECTORA  🗳️</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size:18px;'>Sri Lanka's AIO Election Platform  </p>",
        unsafe_allow_html=True
    )

    # Create a div container with the class name 'white-space-bg' to set the background image for the text area
    st.markdown(
        """
       <div class='white-space-bg'>
        <div class='blurred-box'>
            <h2 style='text-align: center; color: black;'><b>Please follow the below steps to use the ELECTORA. ⬇️</b></h2>
            <ol style='margin: 0 auto; max-width: 600px; color: black;'>
                <li><b>Click on the <b>Query/Compare Manifestos</b> button 📜 to upload the manifestos of different political parties and ask questions about them.</b></li>
                <li><b>Click on the <b>Chat with Agent</b> button 🎤 to report problems in your district and see full analytic report.</b></li>
                <li><b>Click on the <b>Predict Win</b> button 🏆 to predict the winning party based on the current situation.</b></li>
            </ol>
        </div>
    </div>
        """,
        unsafe_allow_html=True
    )

    # Add a separator line
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)
    
    chat_with_agent()
    col1, col2, col3,col4 = st.columns([1, 1, 1,1])

    # Place buttons in each column with customized styling
    with col1:
        if st.button("📜 Query/Compare Manifestos ", key="Query Manifiesto", use_container_width=True):
            st.session_state.current_function = "query_manifiesto"

    with col2:
        if st.button("🎤U Reporter", key="U reporter", use_container_width=True):
            st.session_state.current_function = "U reporter"

    with col3:
        if st.button("🏆 Predict Win ", key="Predict Win", use_container_width=True):
            st.session_state.current_function = "win predict"
    
    with col4:
        if st.button("📰 Latest News", key="Latest News", use_container_width=True):
            st.session_state.current_function = "latest news"

# Execute the main function
if __name__ == "__main__":

    if "current_function" not in st.session_state:
        st.session_state.current_function = 'main'
    if st.session_state.current_function == "main":
        main()
    elif st.session_state.current_function == "query_manifiesto":
        query_manifiesto()
    elif st.session_state.current_function == "U reporter":
        fb.reporter()
    elif st.session_state.current_function == "win predict":
        win_predict()
    elif st.session_state.current_function == "latest news":
        electionNews()