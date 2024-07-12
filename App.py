import re
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import pandas as pd

# preprocessing text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()

    # removing urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # removing HTML
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r'', text)

    # removing punctuations
    punctuation_pattern = re.compile(r'[^\w\s]')
    text = punctuation_pattern.sub(r'', text)

    # tokenization
    text = word_tokenize(text)

    # removing stopwords
    stop_words_set = set(stopwords.words("english"))
    text = [word for word in text if word not in stop_words_set]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    # sticking sentence back and returning
    return ' '.join(text)


# predicting sentiment
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    features = tfidf_vectorizer.transform([processed_review]).toarray()
    sentiment = model.predict(features)[0]

    if(sentiment == "pos"):
        return "positive"
    elif(sentiment == "neg"):
        return "negative"
    else:
        return "neutral"

# bulk prediction
def predict_sentiments_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return []
    
    # storing predicted sentiments
    predicted_sentiments = []
    
    for review in df['Review']:
        sentiment = predict_sentiment(review)
        predicted_sentiments.append(sentiment)
    
    return predicted_sentiments

# determining overall sentiment of the course
def overall_sentiment(sentiments):
    sentiment_counts = Counter(sentiments)
    total_reviews = len(sentiments)
    positive_percentage = (sentiment_counts['positive'] / total_reviews) * 100
    
    if positive_percentage > 50:
        return 'positive'
    elif sentiment_counts['negative'] > sentiment_counts['positive']:
        return 'negative'
    else:
        return 'neutral'

# ------------------------------------------ MAIN ---------------------------------------------

# downloading NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# loading model and TfidfVectorizer
model = load('models/logistic_regression_model.joblib')
tfidf_vectorizer = load('models/tfidf_vectorizer.joblib')

# basic UI
print("+---------------------------------------+")
print("|       Emotions Behind the Screen      |")
print("|    Sentiment in Reviews of Courses    |")
print("+---------------------------------------+")
print("| Input                                 |")
print("| 1. Analyze single review              |")
print("| 2. Analyze csv file(bulk review)      |")
print("+---------------------------------------+")

# user input
user_ask = input(">>> ")

# single review
if(user_ask == "1"):
    user_input = input("Enter a review: ")
    try:
        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment is: {sentiment}")
    except Exception as e:
        print("Some error occured")

# bulk review
elif(user_ask == "2"):
    csv_path = input("Enter path of csv file: ")
    try:
        predicted_sentiments = predict_sentiments_from_csv(csv_path)
        overall_sentiment = overall_sentiment(predicted_sentiments)
        print(f"The overall sentiment of the course is: {overall_sentiment}")
    except Exception as e:
        print("Some error occured, perhaps a wrong path to csv file")

# wrong input
else:
    print("Incorrect Input")