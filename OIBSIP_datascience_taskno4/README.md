# Objective

The objective of this project is to build an Email Spam Detector using Machine Learning in Python. Spam emails are one of the biggest problems on the internet, often used for phishing, scams, and spreading malicious content. This project trains a model to classify emails as Spam or Not Spam based on their content.

# Steps Involved

Dataset Collection
Used the popular Spam Email dataset (spam.csv) containing labeled examples of spam and ham emails.

Data Preprocessing

Removed unnecessary columns.

Encoded labels: ham → 0, spam → 1.

Cleaned and prepared email text messages.

Feature Extraction

Used TF-IDF Vectorization to convert text data into numerical features that machine learning models can understand.

Model Training

Applied Logistic Regression to train the spam classifier.

Logistic Regression was chosen for its effectiveness in binary classification problems like spam detection.

Model Evaluation

Evaluated performance using Accuracy, Confusion Matrix, Precision, Recall, and F1-score.

Compared predictions against actual labels.

Testing with Custom Inputs

Tested model with new email text samples to check predictions in real scenarios.


 # Tools & Technologies Used

Python 3 – Programming language.

pandas – Data handling and preprocessing.

scikit-learn – Machine learning (Logistic Regression, train-test split, evaluation metrics).

TfidfVectorizer – For feature extraction from email text.

VS Code / Jupyter Notebook – Development environment.

# Outcome

Successfully trained a machine learning model to detect spam emails.

Achieved high accuracy and reliable classification between spam and non-spam messages.

The model can be extended and deployed in real-world applications like email filters or chat spam detectors.
