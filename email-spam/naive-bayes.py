import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split

def remove_punctuation_and_lower(text):
    return "".join([char for char in text if char not in string.punctuation]).lower()

def remove_stopwords(word_vector):
    STOP_WORDS = nltk.corpus.stopwords.words('english')
    return [word for word in word_vector if word not in STOP_WORDS][1:]

def tokenize(text):
    return text.split()


class NaiveBayes:
    # Returns the probability of the words given that it's spam or not spam
    def __init__(self, min_word_occurance):
      self.MIN_WORD_OCCURANCE = min_word_occurance

    def P_words(self, text_list, labels):
      if(len(text_list) != len(labels)):
        raise Exception("Inputs not same length")

      dict_prob = {} # For each word in vocabulary consist P(word|!spam) and P(word|spam)

      # Count number of words for both spam and not spam
      for i in range(len(text_list)):
        text = text_list[i][0] # wtf is this, TODO: fix it
        label = labels[i]
        is_spam = True if label == 1 else False

        words = text.split()
        for word in words:
          # First time seeing this word
          if word not in dict_prob:
            dict_prob[word] = {"spam": 0, "not_spam": 0}

          dict_prob[word]['spam' if is_spam else 'not_spam'] += 1

      # Calculating probability
      # print(dict_prob)

      for key in dict_prob:
        total = dict_prob[key]['spam'] + dict_prob[key]['not_spam']
        # TODO: Check diff values
        # if(total < self.MIN_WORD_OCCURANCE): # Ignore words which have little amount of occurance
          # del dict_prob[key]
        # else:
        dict_prob[key]['spam_prob'] = dict_prob[key]['spam'] / total
        dict_prob[key]['not_spam_prob'] = dict_prob[key]['not_spam'] / total
        del dict_prob[key]['spam']
        del dict_prob[key]['not_spam']

      print(dict_prob)
      return dict_prob

    def predict(self, X_test, y_test):
      pass

    def fit(self, X_train, y_train):
      self.X_train = X_train
      self.y_train = y_train

      # Building a prio
      _, counter = np.unique(y_train, return_counts=True)
      self.non_spam_prob = counter[1] / len(y_train)
      self.spam_prob = counter[0] / len(y_train)

      # Creating the Bag of Words model
      self.dict_prob = self.P_words(X_train, y_train)


def main():
    data = pd.read_csv('./emails.csv')
    # print(data.head())

    # Data cleaning
    data = data.drop_duplicates(keep = 'last')

    data['text'] = data['text'].apply(lambda text:remove_punctuation_and_lower(text))
    data['text'] = data['text'].apply(lambda text:tokenize(text))
    data['text'] = data['text'].apply(lambda text:remove_stopwords(text))
    data['text'] = data['text'].apply(lambda text: " ".join(text))

    X = data.drop(columns=['spam'])
    y = data.spam
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    nb = NaiveBayes(3)
    nb.fit(X_train.to_numpy(), y_train.to_numpy())
    nb.predict(X_test, y_test)


if __name__ == "__main__":
    main()
