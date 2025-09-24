# IMDb Sentiment Analysis - First Year Task
# Author: Your Name

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\Siddharth\OneDrive\Desktop\PYTHON PROGRAMMS\IMDB_Dataset.csv.csv")

# 2. Preprocessing
# Convert text to lowercase (CountVectorizer handles tokenization & stopwords)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# 5. Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)

# 6. Results
print("Model Accuracies:")
print(f"Logistic Regression: {acc_lr:.4f}")
print(f"Naive Bayes: {acc_nb:.4f}")
