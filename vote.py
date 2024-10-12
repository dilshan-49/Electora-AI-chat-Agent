import firebase as fb
import streamlit as st

db = fb.db

# Function to increment the count for a specific button
def increment_vote_count(party_name):
    # Reference to the specific party document
    doc_ref = db.collection('votes').document(party_name)
    
    try:
        doc = doc_ref.get()
        if doc.exists:
            current_vote = doc.to_dict().get('vote', 0)
            new_vote = current_vote + 1
            # Update the vote count
            doc_ref.update({'vote': new_vote})
        else:
            # If the document doesn't exist, initialize it with 1 vote
            doc_ref.set({'vote': 1})
    except Exception as e:
        st.error(f"Error updating vote count: {e}")

# Function to get button counts from Firestore
def get_vote_counts():
    parties = ['NPP', 'SJB', 'UNP','SLPP']
    votes = {}
    
    try:
        for party in parties:
            doc_ref = db.collection('votes').document(party)
            doc = doc_ref.get()
            if doc.exists:
                votes[party] = doc.to_dict().get('vote', 0)
            else:
                votes[party] = 0
    except Exception as e:
       return e
    
    return votes


