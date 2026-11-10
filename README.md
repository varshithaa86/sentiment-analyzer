Sentiment Analyzer – Machine Learning Project

1. Project Overview

The Sentiment Analyzer is a machine learning-based application that analyzes textual reviews and classifies them into two sentiment categories: Positive and Negative.

The project uses Natural Language Processing (NLP) techniques to process text and a Logistic Regression classification model to predict the sentiment of user reviews.

An interactive web interface is developed using Streamlit, allowing users to enter reviews and receive real-time sentiment predictions.

2. Objectives

* To understand the fundamentals of Natural Language Processing.
* To classify customer reviews based on their sentiment.
* To implement a machine learning classification algorithm.
* To develop an interactive application using Streamlit.
* To evaluate the performance of the trained model.

3. Technologies Used

Technology	Purpose
Python	Programming language
Pandas	Dataset handling
Scikit-learn	Machine learning
TF-IDF	Text feature extraction
Logistic Regression	Sentiment classification
Joblib	Model serialization
Streamlit	Web application interface

4. Features

* Accepts user reviews as text input.
* Predicts Positive or Negative sentiment.
* Uses a trained machine learning model.
* Provides an interactive Streamlit interface.
* Displays sentiment results clearly.
* Supports real-time review analysis.

5. Project Structure

sentiment-analyzer/
│
├── app.py
├── train_model.py
├── requirements.txt
├── sentiment_model.pkl
└── README.md

File descriptions:

* app.py – Runs the Streamlit application and displays sentiment predictions.
* train_model.py – Prepares the training data, trains and evaluates the machine learning model.
* requirements.txt – Contains the required Python libraries.
* sentiment_model.pkl – Stores the trained machine learning model.

6. Machine Learning Algorithms

TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF converts textual reviews into numerical features that can be processed by a machine learning algorithm. It gives importance to words based on their frequency and distribution across documents.

Logistic Regression

Logistic Regression is a supervised machine learning classification algorithm used to predict whether a review belongs to the Positive or Negative category.

7. System Workflow

1. Collect sample customer reviews.
2. Assign sentiment labels to the reviews.
3. Split the dataset into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train the Logistic Regression model.
6. Evaluate the trained model.
7. Save the trained model using Joblib.
8. Load the model in the Streamlit application.
9. Accept a new review and predict its sentiment.

8. Installation and Execution

Prerequisites

* Python 3.10 or above
* Visual Studio Code
* Internet connection for installing dependencies

Step 1: Clone the repository

git clone https://github.com/varshithaa86/sentiment-analyzer.git

Step 2: Open the project folder

cd sentiment-analyzer

Step 3: Install dependencies

pip install -r requirements.txt

Step 4: Train the model

python train_model.py

Step 5: Run the application

streamlit run app.py

The application will open in your browser.

9. Sample Predictions

Sample Review	Expected Sentiment
This product is amazing	Positive
I love this product	Positive
Excellent quality and service	Positive
The service was disappointing	Negative
This product is terrible	Negative
I regret buying this	Negative

Actual predictions depend on the trained model and its learned features.

10. Applications

* Customer feedback analysis
* Product review classification
* E-commerce review analysis
* Service quality monitoring
* Social media sentiment analysis

11. Limitations

* The model uses a limited sample dataset.
* Predictions may be less accurate for complex sentences, sarcasm, and mixed sentiments.
* Model performance depends on the quality and quantity of training data.

12. Future Enhancements

* Train the model using a larger real-world review dataset.
* Support neutral sentiment classification.
* Improve accuracy through model optimization.
* Add graphical visualizations and sentiment statistics.
* Support multiple languages.

13. Conclusion

The Sentiment Analyzer demonstrates how machine learning and Natural Language Processing can be used to classify textual reviews. By combining TF-IDF, Logistic Regression, and Streamlit, the project provides an interactive application for analyzing customer opinions.

14. Author

Project: Sentiment Analyzer
Domain: Machine Learning and Natural Language Processing
Language: Python
Interface: Streamlit
