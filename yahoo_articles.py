import requests
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load pre-trained BERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def get_yahoo_finance_articles(base_url, count=26):
    """Retrieves Yahoo Finance articles from the specified base URL."""
    response = requests.get(base_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='js-stream-content')
        result = []
        for article in articles[:count]:
            title_tag = article.find('h3')
            link_tag = article.find('a')
            if title_tag and link_tag:
                title = title_tag.get_text(strip=True)
                link = link_tag['href']
                if not link.startswith('http'):
                    link = 'https://finance.yahoo.com' + link
                result.append({"title": title, "link": link})
        return result
    return []

def extract_text_from_article(link):
    """Extracts text from a given article link."""
    response = requests.get(link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.text.lower() for p in paragraphs])
    return ""

def parse_all_articles(links):
    """Parses all articles and returns a list of their text content."""
    return [extract_text_from_article(link) for link in links]

def data_preprocessing(texts):
    """Preprocesses text using tokenization and lemmatization."""
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(token) for token in sent_tokenize(doc)]) for doc in texts]

def get_best_article(user_input, preprocessed_bdd):
    """Finds the most similar article to the user input."""
    all_texts = preprocessed_bdd + [user_input]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_similar_index = np.argmax(cosine_sim)
    return preprocessed_bdd[most_similar_index] if 0.1 < cosine_sim[0][most_similar_index] < 1 else "I am sorry, I could not understand you."

def start_chatbot_yahoo(preprocessed_bdd):
    """Starts an interactive chatbot session."""
    print("Type 'q', 'quit', or 'exit' to leave the chatbot.")
    while True:
        query = input("\nUser: ")
        if query.lower() in {"q", "quit", "exit", "bye"}:
            break
        best_article = get_best_article(query, preprocessed_bdd)
        print(f"\nChatbot: {best_article}")

def get_financial_advices():
    """Main function to retrieve articles and start the chatbot."""
    print("Chatbot: Fetching financial news articles...")
    article_sources = [
        "https://finance.yahoo.com/topic/personal-finance-news/",
        "https://finance.yahoo.com/",
        "https://finance.yahoo.com/calendar/",
        "https://finance.yahoo.com/topic/stock-market-news/"
    ]
    articles = sum([get_yahoo_finance_articles(url) for url in article_sources], [])
    links = [article['link'] for article in articles]
    bdd = parse_all_articles(links)
    preprocessed_bdd = data_preprocessing(bdd)
    start_chatbot_yahoo(preprocessed_bdd)

if __name__ == "__main__":
    get_financial_advices()
