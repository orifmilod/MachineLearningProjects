import pandas as pd
import numpy as np
import string
from nltk.corpus import words
import nltk
import math
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

# https://en.wikipedia.org/wiki/Additive_smoothing
A = 1
D = 2

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
      self.word_set = set(words.words())

    def build_vocabulary(self, text_list, labels):
      if(len(text_list) != len(labels)):
        raise Exception("Inputs not same length")

      word_counter_dict = {}

      # Count number of words for both spam and not spam
      for i in range(len(text_list)):
        text = text_list[i][0] # TODO: fix this garbage
        label = labels[i]
        is_spam = True if label == 1 else False

        words = text.split()
        for word in words:
            # Not an english word
            # if word not in self.word_set:
              # continue

            # First time seeing this word
            if word not in word_counter_dict:
              word_counter_dict[word] = {"spam": 0, "not_spam": 0} # Laplace smoothing with alpha=1 

            word_counter_dict[word]['spam' if is_spam else 'not_spam'] += 1

      return word_counter_dict

    def predict(self, X_test, y_test):
      correct = 0

      for i in range(len(X_test)):
        text = X_test[i][0] # TODO: fix this garbage
        label = y_test[i]
        # print(text)

        spam_prob =  0 #math.log(self.spam_prob_prio, math.e)
        not_spam_prob = 0 #math.log(self.non_spam_prob_prio, math.e)
        words = text.split()

        for word in words:
            if(word in self.word_counter_dict):
                # Taking natural log of probabilities to it does got to 0
                prob = (self.word_counter_dict[word]['spam'] + A) / (self.spam_counter + A * D)
                # print("Word:", word)
                # print(prob)
                spam_prob += math.log(prob, math.e)

                prob = (self.word_counter_dict[word]['not_spam'] + A) / (self.non_spam_counter + A * D)
                not_spam_prob += math.log(prob, math.e)
            else:
                prob = A / (A * D)
                spam_prob += math.log(prob, math.e)
                not_spam_prob += math.log(prob, math.e)

        spam_prob = spam_prob / (spam_prob + not_spam_prob) 
        not_spam_prob = not_spam_prob / (spam_prob + not_spam_prob)

        print("Spam prob", spam_prob / (spam_prob + not_spam_prob))
        print("Not spam prob", not_spam_prob / (spam_prob + not_spam_prob))
        print("Label:", label)
        is_spam = spam_prob > not_spam_prob

        if(label == 1 and is_spam):
            correct += 1

      print(correct / len(y_test))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # Building a prio
        _, counter = np.unique(y_train, return_counts=True)
        self.spam_counter = counter[1]
        self.non_spam_counter = counter[0]

        self.spam_prob_prio = counter[1] / len(y_train)
        self.non_spam_prob_prio = counter[0] / len(y_train)

        print(self.spam_counter, self.non_spam_counter)

        # Creating the Bag of Words model
        self.word_counter_dict = self.build_vocabulary(X_train, y_train)
        self.num_of_spam_words = 0
        self.num_of_non_spam_words = 0

        for key in self.word_counter_dict:
          self.num_of_non_spam_words += self.word_counter_dict[key]['not_spam']
          self.num_of_spam_words += self.word_counter_dict[key]['spam']



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
    nb.predict(X_test.to_numpy(), y_test.to_numpy())


if __name__ == "__main__":
    main()
