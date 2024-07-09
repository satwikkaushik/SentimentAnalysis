# Logistic Regression Model for Sentiment Analysis

This repository contains a machine learning project that utilizes a Logistic Regression model to classify students' review, of an online course, based on various features. The project aims to demonstrate the application of logistic regression in binary classification problems, model evaluation techniques, and the use of confusion matrices for performance visualization.
The output of the analysis is devided into three category: Positive, Negative and Neutral
The project is also able to analye reviews in bulk by reading it from a csv file.

## Project Structure

- `Training.ipynb` : Contains the code for training the Logistic Regression model using scikit-learn.
- `App.py` : Entry point of the project, contains basic UI for user input and result output

## Model

The project uses a Logistic Regression model from scikit-learn for binary classification. Logistic Regression is chosen for its simplicity and effectiveness in binary classification tasks.
The model is trained on a dataset containing 1,40,000+ reviews from coursera, thus producing results with an accuracy of 93%.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
