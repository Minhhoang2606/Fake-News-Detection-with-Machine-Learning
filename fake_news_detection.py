'''
Fake News Detection using Machine Learning
Author: Henry Ha
'''
# Import the libraries
import pandas as pd

#TODO Exploratory Data Analysis (EDA)

# Load the training dataset
train_data = pd.read_csv('train.csv')

# Display dataset information
print(train_data.info())

# Display the first few rows
print(train_data.head())

# Count the occurrences of each label
label_counts = train_data['label'].value_counts()

# Plot the distribution
import matplotlib.pyplot as plt

label_counts.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribution of Labels')
plt.xlabel('Label (0: Real, 1: Fake)')
plt.ylabel('Count')

# Add the value of each bar on top of it
for p in plt.gca().patches:
    plt.text(p.get_x() * 1.005, p.get_height() * 1.005,
             '{:.0f}'.format(p.get_height()), fontsize=12, color='black')
plt.show()

# Compute lengths of title and text
train_data['title_length'] = train_data['title'].str.len()
train_data['text_length'] = train_data['text'].str.len()

# Plot title and text length distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
train_data['title_length'].hist(ax=axes[0], bins=20, color='blue', alpha=0.7)
axes[0].set_title('Title Length Distribution')
train_data['text_length'].hist(ax=axes[1], bins=20, color='green', alpha=0.7)
axes[1].set_title('Text Length Distribution')
plt.tight_layout()
plt.show()

from wordcloud import WordCloud

# Generate word clouds for fake and real news
fake_news_text = ' '.join(train_data[train_data['label'] == 1]['text'].dropna())
real_news_text = ' '.join(train_data[train_data['label'] == 0]['text'].dropna())

fig, ax = plt.subplots(1, 2, figsize=(15, 7))

fake_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(fake_news_text)
ax[0].imshow(fake_wordcloud, interpolation='bilinear')
ax[0].set_title('Word Cloud: Fake News')
ax[0].axis('off')

real_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(real_news_text)
ax[1].imshow(real_wordcloud, interpolation='bilinear')
ax[1].set_title('Word Cloud: Real News')
ax[1].axis('off')
plt.show()

# Count missing values
missing_values = train_data.isnull().sum()

# Display missing data information
print(missing_values)

#TODO Text Preprocessing

# Replace missing values in textual columns with empty strings
train_data['title'] = train_data['title'].fillna('')
train_data['author'] = train_data['author'].fillna('')
train_data['text'] = train_data['text'].fillna('')

# Combine title, author, and text into one column
train_data['content'] = train_data['title'] + ' ' + train_data['author'] + ' ' + train_data['text']

from nltk.tokenize import word_tokenize

# Tokenize and lowercase the text
train_data['content'] = train_data['content'].apply(lambda x: ' '.join(word_tokenize(x.lower())))

from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))

# Remove stopwords and punctuation
train_data['content'] = train_data['content'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in stop_words and word not in string.punctuation]
))

from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Apply stemming
train_data['content'] = train_data['content'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the text data
X = tfidf_vectorizer.fit_transform(train_data['content']).toarray()
y = train_data['label']

#TODO Building the model

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Calculate accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Display detailed metrics
print(classification_report(y_val, y_val_pred))

#TODO Testing the Model on Kaggle test.csv

# Load the test data
test_data = pd.read_csv('test.csv')

# Handle missing values
test_data['title'] = test_data['title'].fillna('')
test_data['author'] = test_data['author'].fillna('')
test_data['text'] = test_data['text'].fillna('')

# Combine title, author, and text
test_data['content'] = test_data['title'] + ' ' + test_data['author'] + ' ' + test_data['text']

# Transform the content into numerical features using the trained TF-IDF vectorizer
X_test = tfidf_vectorizer.transform(test_data['content']).toarray()

# Predict labels for test data
y_test_pred = model.predict(X_test)

import numpy as np

# Check the distribution of predicted labels
unique, counts = np.unique(y_test_pred, return_counts=True)
print("Predicted class distribution:", dict(zip(unique, counts)))

# Analyze predictions based on text length
test_data['text_length'] = test_data['content'].str.len()
test_data['predicted_label'] = y_test_pred

# Compare text length distribution for each predicted class
test_data.groupby('predicted_label')['text_length'].mean()

# Display a few sample predictions with text content
for i in range(5):  # Adjust number as needed
    print(f"Sample {i+1}:")
    print("Content:", test_data['content'].iloc[i][:200])  # Limit to first 200 characters
    print("Predicted Label:", y_test_pred[i])
    print()
