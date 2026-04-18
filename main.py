import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read data
data = pd.read_csv("expenses.csv")

X = data["Description"]
y = data["Category"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vectors, y)

# User input
expense = input("Enter expense name: ")

# Predict
expense_vector = vectorizer.transform([expense])
prediction = model.predict(expense_vector)

print("Predicted Category:", prediction[0])