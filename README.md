p# Helsinki-AI-Project

# A simple conceptual demonstration of the AI classification approach
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training data (In a full project, this would be a massive dataset)
training_articles = [
    "Scientists discover new planet orbiting a distant star.",
    "Shocking secret! Aliens landed in downtown tomorrow morning!",
    "The government passes new economic reform bill today.",
    "You won't believe how this fruit cures all diseases instantly!"
]
# 1 = Real News, 0 = Suspicious/Fake
labels = [1, 0, 1, 0] 

# Vectorizing the text (Turning text into a matrix of TF-IDF features)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_articles)

# Using Logistic Regression (a core technique from Chapter 3 & 4)
model = LogisticRegression()
model.fit(X_train, labels)

# Testing the model on a brand new article
new_article = ["Unbelievable medical miracle hidden by doctors for years!"]
X_new = vectorizer.transform(new_article)
prediction = model.predict(X_new)

# Outputting perspective and prediction
print(f"Analyzing article: '{new_article[0]}'")
if prediction[0] == 0:
    print("AI Assessment: Warning! This text contains high-risk sensationalist patterns.")
else:
    print("AI Assessment: Looks normal.")
# Final Project: Fake News & Misinformation Detector
​This project is an AI-powered text analysis tool designed to identify and flag potentially misleading or fake news articles. By utilizing Natural Language Processing (NLP) and machine learning classification techniques (specifically TF-IDF and Logistic Regression), the tool analyzes linguistic patterns, emotional tones, and clickbait structures to assign a "credibility score" to written content. The ultimate goal is to empower internet users with a lightweight browser extension that helps combat misinformation and promotes digital literacy.


