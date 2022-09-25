import pandas as pd
import numpy as np
import string
from nltk.corpus import words
import nltk
import math
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer

def stemText(text):
  ps = PorterStemmer()
  z = []
  for i in text:
      z.append(ps.stem(i))
  w = z[:]
  z.clear()
  return w

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def remove_punctuation_and_lower(text):
    return "".join([char for char in text if char not in string.punctuation]).lower()

def remove_stopwords(word_vector):
    STOP_WORDS = nltk.corpus.stopwords.words('english')
    return [word for word in word_vector if word not in STOP_WORDS][1:]

def tokenize(text):
    return text.split()


class LogisticRegression:
    def __init__(self):
        print("init")

    def fit(self, datas, labels):
      pass

    def predict(self, datas, labels):
      pass



def main():
    data = pd.read_csv('./emails.csv')

    # Data cleaning
    data = data.drop_duplicates(keep = 'last')

    data['text'] = data['text'].apply(lambda text:remove_punctuation_and_lower(text))
    data['text'] = data['text'].apply(lambda text:tokenize(text))
    data['text'] = data['text'].apply(lambda text:remove_stopwords(text))
    data['text'] = data['text'].apply(stemText)
    data['text'] = data['text'].apply(lambda text: " ".join(text))

    X = data.text
    y = data.spam

    training_data, test_data, training_label, test_label = train_test_split(X, y, test_size = 0.2, random_state = 42)

    nb = NaiveBayes(3)
    nb.fit(training_data.to_numpy(), training_label.to_numpy())
    nb.predict(test_data.to_numpy(), test_label.to_numpy())


if __name__ == "__main__":
    main()

