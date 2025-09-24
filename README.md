# IMDb Sentiment Analysis - First Year AIML Recruitment Task

# Folder Structure
imdb.py
IMDB_Dataset.csv
requirements.txt


# Dataset
- Dataset: IMDb Movie Review Dataset (50,000 reviews)
- Download from: [Kaggle Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Place `IMDB_Dataset.csv` in the project folder before running the script.
- Columns:
  - **review** → the movie review text
  - **sentiment** → label (`positive` / `negative`)

# Approach
1. Loaded the dataset into a pandas DataFrame.
2. Preprocessed text using **CountVectorizer**:
   - Converted text to lowercase
   - Removed English stopwords
3. Split the dataset into **80% training / 20% testing**.
4. Trained two models:
   - **Logistic Regression**
   - **Naive Bayes**
5. Evaluated both models using **accuracy** metric.

# Results
- Logistic Regression Accuracy: **88.27%**
- Naive Bayes Accuracy: **85.66%**

Logistic Regression performed slightly better, so it is the preferred model.

# How to Run
1. Install required libraries:
   pip install -r requirements.txt
2. Run:
   python imdb.py
