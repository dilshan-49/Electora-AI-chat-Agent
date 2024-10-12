import datetime
import spacy
from firebase_admin import firestore


def fetch_latest_articles(db):
    '''Fetch the 10 most recent news articles from the Firestore database'''

    # Query for articles added in the last 5 days
    articles_ref = db.collection('news')
    recent_articles = articles_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(10).stream()
    return [article.to_dict() for article in recent_articles]

def get_relevant_news(request,db):
    '''Fetch news articles that contain any of the keywords extracted from the user's query'''

    news_ref = db.collection('news')
    #following part is commented as our news database is low right now. Therefore we will fetch all news articles straight to our AI agent
    '''
    nlp = spacy.load("en_core_web_sm")
    doc=nlp(request)
    keywords = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN','nsubj', 'dobj', 'pobj','ADJ')]
    keywords = list(set(keywords))  # Remove duplicates


    # Query for documents where the 'keywords' field contains any of the keywords from the keyword_list
    query = news_ref.where('keywords', 'array_contains_any', keywords)
    '''
    results = news_ref.stream()

    # Create a list to hold the matched documents
    matched_articles = []
    for doc in results:
        matched_articles.append(doc.to_dict())  # Append the document data to the list
    return matched_articles

