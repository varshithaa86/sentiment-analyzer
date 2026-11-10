# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Dataset: review and sentiment
data = [
    # Positive reviews
    ("I love this product", "Positive"),
    ("I like this product", "Positive"),
    ("This product is amazing", "Positive"),
    ("Very good quality", "Positive"),
    ("I am happy with this product", "Positive"),
    ("Excellent product", "Positive"),
    ("Wonderful experience", "Positive"),
    ("Best purchase ever", "Positive"),
    ("The product is fantastic", "Positive"),
    ("I am very satisfied", "Positive"),
    ("This is a great product", "Positive"),
    ("The quality is excellent", "Positive"),
    ("I enjoyed using it", "Positive"),
    ("Absolutely wonderful", "Positive"),
    ("Very useful product", "Positive"),
    ("I recommend this product", "Positive"),
    ("This product is awesome", "Positive"),
    ("The service was excellent", "Positive"),
    ("I am pleased with my purchase", "Positive"),
    ("Everything is perfect", "Positive"),
    ("Good experience", "Positive"),
    ("Worth buying", "Positive"),
    ("I am very happy", "Positive"),
    ("The product works perfectly", "Positive"),
    ("I really like it", "Positive"),
    ("This is a wonderful product", "Positive"),
    ("I am satisfied with this product", "Positive"),
    ("The product is very good", "Positive"),
    ("I would recommend it", "Positive"),
    ("Great service and quality", "Positive"),

    # Negative reviews
    ("I hate this product", "Negative"),
    ("I do not like this product", "Negative"),
    ("I don't like this product", "Negative"),
    ("I dont like this product", "Negative"),
    ("This product is terrible", "Negative"),
    ("Very bad quality", "Negative"),
    ("I am unhappy", "Negative"),
    ("Worst product ever", "Negative"),
    ("I dislike this product", "Negative"),
    ("Very disappointing", "Negative"),
    ("Bad experience", "Negative"),
    ("I regret buying this", "Negative"),
    ("The product is awful", "Negative"),
    ("I am very dissatisfied", "Negative"),
    ("This is not good", "Negative"),
    ("The quality is poor", "Negative"),
    ("Not worth the money", "Negative"),
    ("I am not satisfied", "Negative"),
    ("The product is damaged", "Negative"),
    ("I would not recommend it", "Negative"),
    ("Very poor experience", "Negative"),
    ("This product is useless", "Negative"),
    ("I am disappointed", "Negative"),
    ("Terrible service", "Negative"),
    ("Waste of money", "Negative"),
    ("I am not happy with this product", "Negative"),
    ("I do not recommend this product", "Negative"),
    ("The product does not work", "Negative"),
    ("I never liked this product", "Negative"),
    ("This is a horrible product", "Negative")
]

# Separate reviews and labels
reviews = [item[0] for item in data]
sentiments = [item[1] for item in data]

# Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 3))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    reviews,
    sentiments,
    test_size=0.20,
    random_state=42,
    stratify=sentiments
)

# Train a temporary model for evaluation
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Training and Evaluation")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Train final model using all available examples
model.fit(reviews, sentiments)

# Save the final trained model
joblib.dump(model, "sentiment_model.pkl")

print("\nFinal model trained on all examples.")
print("Model saved successfully!")