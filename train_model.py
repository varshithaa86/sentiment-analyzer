
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Balanced sample dataset
data = {
    "text": [
        "This product is excellent",
        "I love this product",
        "Amazing quality and great service",
        "Very happy with my purchase",
        "The product is wonderful",
        "I am satisfied with this item",
        "Absolutely fantastic experience",
        "This is a great product",
        "I really like it",
        "Excellent value for money",
        "The quality is outstanding",
        "Delivery was fast and smooth",
        "I highly recommend this product",
        "The service was very helpful",
        "This exceeded my expectations",
        "Beautiful design and perfect quality",
        "Really pleased with the results",
        "The experience was enjoyable",
        "Works perfectly and looks great",
        "I would buy this again",
        "This product is terrible",
        "I hate this product",
        "Very poor quality",
        "I am unhappy with my purchase",
        "The product is awful",
        "Very disappointed",
        "Worst experience ever",
        "This is a bad product",
        "I really dislike it",
        "Waste of money",
        "The quality is horrible",
        "Delivery was very slow",
        "I do not recommend this product",
        "The service was disappointing",
        "This failed my expectations",
        "The design is ugly and cheap",
        "I regret buying this item",
        "The experience was frustrating",
        "It stopped working quickly",
        "I would not buy this again",
        "The product is reliable",
        "The app is easy to use",
        "Customer support was excellent",
        "The results were impressive",
        "I am delighted with the purchase",
        "Everything works as expected",
        "The packaging was beautiful",
        "This is worth the price",
        "The product feels premium",
        "I am very pleased",
        "The product broke immediately",
        "Customer support was unhelpful",
        "The app is confusing",
        "The results were disappointing",
        "I regret this purchase",
        "Nothing works properly",
        "The packaging was damaged",
        "This is not worth the price",
        "The product feels cheap",
        "I am very dissatisfied"
    ],
    "sentiment": [
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative"
    ]
}

df = pd.DataFrame(data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=df["sentiment"]
)

# Build the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Predict on unseen test examples
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test, y_pred, zero_division=0
)
matrix = confusion_matrix(
    y_test, y_pred, labels=["Negative", "Positive"]
)

# Display results
print("=" * 45)
print("       SENTIMENT ANALYZER")
print("=" * 45)
print("Total reviews:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(report)

print("Confusion Matrix")
print("Labels: Negative, Positive")
print(matrix)

print("\nSample Predictions:")
test_reviews = [
    "This product is amazing and wonderful",
    "This product is horrible and terrible",
    "I am very happy with the quality",
    "I regret buying this",
    "The quality is excellent",
    "The quality is awful"
]

for review in test_reviews:
    prediction = model.predict([review])[0]
    print(f"Review: {review}")
    print(f"Sentiment: {prediction}\n")
