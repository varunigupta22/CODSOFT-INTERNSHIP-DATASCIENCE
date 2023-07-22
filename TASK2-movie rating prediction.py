import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('movies.txt')
df = df.dropna()
reviews = df['Review']
ratings = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

y_pred = classifier.predict(X_test_vectors)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)





