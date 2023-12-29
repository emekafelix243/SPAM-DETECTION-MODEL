SPAM DECTECTION MODEL
Author : Okeke Felix Emeka
Email : okeke243@gmail.com	linkedin : FELIX EMEKA

DOMAIN BACKGROUND

This project aims to implement a machine learning model for training a spam detection model using various classifiers. The model is trained on a dataset containing mails that contains spam and ham, with the goal of identifying patterns associated with spam and ham mails and to develop a sophisticated system for the detection of mails that could be classified as spam or ham.
•	This dataset has been collected from free or free for research sources at the Internet.
•	The collection is composed of just one text file, where each line has the correct class followed by the raw message.

PROBLEM STATEMENT

In the paper, we would try to analysis different methods to identify spam messages. We will use the different approach, based on word count and term-frequency inverse document-frequency (tf-idf) transform to classify the messages. Following steps are required in order to achieve the objective:

1.	Download and pre-process the SMS Spam Collection dataset.
2.	Text Vectorization (using TF-IDF) to classify the messages.
3.	Clean the text by removing special characters for better performance
4.	Perform feature engineering techniques: classifying the dataset using label encoder and one hot encoding
5.	Visualized the cleaned text using word cloud, bar charts and box plot for classification purposes
6.	Split the dataset into training and testing sets
7.	Select only the categorical features for X_train_categorical
8.	Initialize various classifier and train them.
9.	Evaluate the classifiers and finding best the model for a dataset.

DATASETS AND INPUTS

The dataset used for this project is SMS Spam Collection Dataset
originates from the Kaggle.com. This dataset has been collected from free or free for research sources at the Internet. The collection is composed of just one text file, where each line has the correct class followed by the raw message. This dataset is comma-separated values (CSV) file. 

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged according being ham (legitimate) or spam.

The files contain one message per line. Each line is composed by two columns: c1 contains the label (ham or spam) and c2 contains the raw text. This corpus has been collected from free or free for research sources at the Internet:

SOLUTION STATEMENT

We are given labelled training data, so this makes it a supervised machine learning problem. For every message, it can be predicted whether it is ham or spam. The accuracy is quantifiable in terms of the f1_score. These performance scores can be compared against the public leader board scores available in the Kaggle website. A well-documented code with dataset will help anyone to replicate the work anywhere on any other machine. 

To begin with I would like to experiment with techniques which we are going to us are based on word count and term-frequency inverse document-frequency (tf-idf) transform. After which I would like to test the approach using many different algorithms like Support Vector Machine, Learning Curves, Decision Tree, AdaBoost, K-Nearest Neighbours and Random Forest, Gradient Boosting Classifier and test the accuracy using f1_score.

BENCHMARK MODEL

Benchmark models are available in Kaggle discussion forums which uses different Machine Learning algorithms. The available public and private leader board score in the Kaggle competition can be used to benchmark the performance of my algorithm. Also, it is possible to explore how the proposed model perform compared to existing models. The result shows that Gradient Boosting Classifier work better on the dataset with an accuracy_score of 0.95.

Evaluation Metrics
Accuracy is the first metric to be checked when the algorithms are evaluated, is the sum of true positives and the true negative outputs divided by the data size. Accuracy means how closer you are to the true value, whereas precision means that your data points are not widely spread. The Scikit-learn library provides a convenience report when working on classification problems to give you a quick idea of the accuracy of a model using a number of matrices, one of them is F1_score which work with all the model.

PROJECT DESIGN
The theoretical workflow of the project would look like:
1.	Download and pre-process the SMS Spam Collection v.1 dataset.
2.	Test and find best approach (tf-idf vectorizer) to classify the messages.
3.	Selection of approach and splitting the dataset into training and testing data.
4.	Initialize various classifier and train it using training data.
5.	Evaluate the classifiers and finding best the model for a dataset using testing data.


