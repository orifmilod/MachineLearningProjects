import pandas as pd
import numpy as np
import string
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def remove_punctuation(text):
    return "".join([char for char in text if char not in string.punctuation]).lower()

def remove_stopwords(word_vector):
    STOP_WORDS = nltk.corpus.stopwords.words('english')
    return [word for word in word_vector if word not in STOP_WORDS][1:]

def tokenize(text):
    return text.split()


class NaiveBayes:
    def predict():
        pass

    def fit():
        pass



def main():
    data = pd.read_csv('./emails.csv')
    # print(data.head())

    # Data cleaning
    data = data.drop_duplicates(keep = 'last')

    data['text'] = data['text'].apply(lambda text:remove_punctuation(text))
    data['text'] = data['text'].apply(lambda text:tokenize(text))
    data['text'] = data['text'].apply(lambda text:remove_stopwords(text))
    data['text'] = data['text'].apply(lambda text: " ".join(text))

    # print(data.head())

    X = data.drop(columns=['spam'])
    y = data.spam

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    corpus = X_train['text'].to_numpy()

    # Uni-gram
    unigram_vectorizer = CountVectorizer(min_df = 50) # Ignoring words occuring less than 50 times 
    C = unigram_vectorizer.fit_transform(corpus)
    print(len(unigram_vectorizer.get_feature_names()))

    #Bi-gram
    bigram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 50)
    C2 = bigram_vectorizer.fit_transform(corpus)
    print(len(bigram_vectorizer.get_feature_names()))

if __name__ == "__main__":
    main()
