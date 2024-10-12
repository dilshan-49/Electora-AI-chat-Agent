import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import plotly.express as px
import nvidia_bot as nv
import json
import pandas as pd

# Initialize Firestore if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate('./ai-electionbot-firebase-adminsdk-h5fv1-fa8b4135c3.json')
    firebase_admin.initialize_app(cred)

# Create a Firestore client
db = firestore.client()

# Function to add a problem to a district
def add_problem_to_district(district_name, problem):
    doc_ref = db.collection('Districts').document(district_name)
    doc = doc_ref.get()

    if doc.exists:
        data = doc.to_dict()
        if 'problems' in data:
            problems = data['problems']
        else:
            problems = []
        problems.append(problem)
        doc_ref.update({'problems': problems})
    else:
        doc_ref.set({'problems': [problem]})

# Function to retrieve all problems for all districts
def get_all_districts_problems():
    districts_ref = db.collection('Districts')
    docs = districts_ref.stream()
    
    all_districts_problems = []
    
    for doc in docs:
        district_name = doc.id
        data = doc.to_dict()
        
        district_data = {
            'district': district_name,
            'problems': data.get('problems', [])
        }
        
        all_districts_problems.append(district_data)

    
    return all_districts_problems

def make_hashable(obj):
    """Recursively convert lists to tuples to make the object hashable."""
    if isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    return obj

def match_districts(geojson, response_dict):
        # Extract the district names from the GeoJSON file
        districts_geojson = [feature['properties']['NAME_1'] for feature in geojson['features']]
        
        # Create a matching dataframe for districts and their respective data from response_dict
        matched_districts = []
        for district in districts_geojson:
            if district in response_dict:
                district_data = response_dict[district]
                # Flatten the dictionary
                for category, weight in district_data.items():
                    matched_districts.append({
                        'district': district,
                        'category': category,
                        'weight': weight
                    })
        df= pd.DataFrame(matched_districts)
        df =df.drop_duplicates()
        return df

def reporter():
    # Streamlit app layout
    st.set_page_config(page_title='District Problem Reporter', page_icon='📰')
    st.markdown("<h1 style='text-align: center; color: #FFA500;'>District Problem Reporter</h1>",
        unsafe_allow_html=True)
    st.markdown(
          """
    <style>
    /* Apply background image to the form section */
    .report-section {
        background-image: url('https://img.freepik.com/premium-vector/journalists-interview-with-business-man-with-cameraman-photographers-tv-show-mass-media-news-blogging-modern-video-report-recent-vector-scene_543062-5540.jpg'); /* Your image URL */
        background-size: cover;
        background-position: center;
        padding: 20px;
        border-radius: 10px;
    }

    /* Style the input fields and buttons */
    textarea, select, input {
        background-color: rgba(255, 255, 255, 0.8); /* Transparent background */
        color: black; /* Text color */
        border-radius: 5px;
        
    }

    

    </style>
    """,
        unsafe_allow_html=True
    )
    
    if st.button("Back to Main Menu 🔙", key="Back to Main Menu"):
        st.session_state.current_function = "main"
    
    
    # Problem Report Section
    st.markdown("""
                <div class="report-section">
                       <h2 style='text-align: center; color: black;'><b>🎤 Report a Problem</b></h2>

                </div>

                 """,

                unsafe_allow_html=True)

                  
                 
    selected_district = st.selectbox('🗺️ Select District', [
        'Ampara', 'Anuradhapura', 'Badulla', 'Batticaloa', 'Colombo', 'Galle', 'Gampaha', 'Hambantota', 'Jaffna',
        'Kalutara', 'Kandy', 'Kegalle', 'Kilinochchi', 'Kurunegala', 'Mannar', 'Matale', 'Matara', 'Monaragala',
        'Mullaitivu', 'Nuwara Eliya', 'Polonnaruwa', 'Puttalam', 'Ratnapura', 'Trincomalee', 'Vavuniya'
    ])
    
   
    problem = st.text_area('Describe the problem')
    
    
    if st.button('🗺️ Submit Problem'):
        if problem:
            add_problem_to_district(selected_district, problem)
            st.success(f'Problem added to {selected_district}')
        else:
            st.error('Please describe the problem before submitting.')

    

    # Generate Report Section
    st.header('Generate Report')
    all_districts_problems = get_all_districts_problems()

    # Convert problem_data to a hashable type


    problem_data = make_hashable(all_districts_problems)

    # Chat Container on the right side
    st.sidebar.header('Chat with reporter Chatbot')
    query = st.sidebar.text_input('Enter your query', key='query')
    if query:
        
        # Generate response using NVIDIA Chatbot
        problem_data = make_hashable(all_districts_problems)
        response = nv.chat_generate(query, problem_data)
        st.sidebar.write(response)

    if st.button('Generate Report', key='generate_report'):
        # Generate report using NVIDIA Chatbot
        
        user_input = "Generate an analytic report on the problems in all districts and generate a report"
        response =nv.generate(user_input, problem_data)
        st.write(response)
        
    response_dict = nv.calc_weight("read whole data provided,use this main categories and create precentage for each districts by counting number of problems under each category.categories:Education,Infrastructure,Economy,Safety and Security,Environment and Agriculture ,Social Welfare.only give output as json dictionary.donot give anything else", problem_data)

    

    # Load the GeoJSON file for Sri Lanka
    with open('./media\gadm41_LKA_1.json\gadm41_LKA_1.json') as f:
        sri_lanka_geojson = json.load(f)

    # Convert the response_dict to a dataframe for plotting
    df = match_districts(sri_lanka_geojson, response_dict)
    st.write("Matched Districts DataFrame:", df)
    # Generate maps for each category
    categories = df['category'].unique()


    for i,category in enumerate(categories):
        # Filter the dataframe for the current category
        category_df = df[df['category'] == category]
        
        # Create the choropleth map
        fig = px.choropleth_mapbox(
            category_df,
            geojson=sri_lanka_geojson,
            locations='district',
            featureidkey="properties.NAME_1",
            color='weight',
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            zoom=6,
            center={"lat": 7.8731, "lon": 80.7718},
            opacity=0.6,
            labels={'weight': f'{category} Problem Percentage'}
        )
        
        # Set the title and display the map
        fig.update_layout(
            title_text=f"{category} Problems by District",
            
            )
        st.plotly_chart(fig, use_container_width=True)

