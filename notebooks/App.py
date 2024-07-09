import re
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import pandas as pd

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Predict sentiment function
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    features = tfidf.transform([processed_review]).toarray()
    sentiment = model.predict(features)[0]
    return sentiment

# Function to read reviews from a CSV and predict sentiment
def predict_sentiments_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return []
    
    # List to store predicted sentiments
    predicted_sentiments = []
    
    # Iterate over each review in the DataFrame
    for review in df['Review']:
        sentiment = predict_sentiment(review)
        predicted_sentiments.append(sentiment)
    
    return predicted_sentiments

# Function to determine the overall sentiment of the course
def determine_overall_sentiment(sentiments):
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

# Load necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the trained model and TfidfVectorizer
model = load('notebooks/sheep_two.joblib')
tfidf = load('notebooks/tfidf_vectorizer.joblib')

print("+---------------------------------------+")
print("|       Emotions Behind the Screen      |")
print("|    Sentiment in Reviews of Courses    |")
print("+---------------------------------------+")
print("| Input                                 |")
print("| 1. Analyze single review              |")
print("| 2. Analyze csv file(bulk review)      |")
print("+---------------------------------------+")

# Take user input
user_ask = input()
if(user_ask == "1"):
    user_input = input("Enter a review: ")
    try:
        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment is: {sentiment}")
    except Exception as e:
        print("Some error occured")

elif(user_ask == "2"):
    csv_path = input("Enter path of csv file: ")
    try:
        predicted_sentiments = predict_sentiments_from_csv(csv_path)
        overall_sentiment = determine_overall_sentiment(predicted_sentiments)
        print(f"The overall sentiment of the course is: {overall_sentiment}")
    except Exception as e:
        print("Some error occured, perhaps a wrong path to csv file")