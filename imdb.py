import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df = pd.read_csv(r"C:\Users\Siddharth\OneDrive\Desktop\PYTHON PROGRAMMS\IMDB_Dataset.csv.csv")
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
print("Model Accuracies:")
print(f"Logistic Regression: {acc_lr:.4f}")
print(f"Naive Bayes: {acc_nb:.4f}")
