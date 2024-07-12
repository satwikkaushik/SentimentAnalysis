# Logistic Regression Model for Sentiment Analysis

This project preprocesses student reviews, extracts features using TF-IDF vectorization, trains a Logistic Regression model, and predicts the sentiment of new reviews. 
The output of the analysis is devided into three category: Positive, Negative and Neutral
It also includes functionalities to analyze a single review or bulk reviews from a CSV file and determine the overall sentiment of a course.

## Project Structure

- `Training.ipynb` : Contains the code for training the Logistic Regression model using scikit-learn.
- `App.py` : Entry point of the project, contains basic UI for user input and result output

## Model

The project uses a Logistic Regression model from scikit-learn for binary classification. Logistic Regression is chosen for its simplicity and effectiveness in binary classification tasks.
The model is trained on a dataset containing 1,40,000+ reviews from coursera, thus producing results with an accuracy of 92%.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
