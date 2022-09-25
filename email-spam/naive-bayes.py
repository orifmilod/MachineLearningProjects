import pandas as pd
import numpy as np
import string
from nltk.corpus import words
import nltk
import math
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer

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
        text = text_list[i]
        label = labels[i]
        is_spam = True if label == 1 else False

        words = text.split()
        for word in words:
            # Not an english word
            # if word not in self.word_set:
              # continue

            # First time seeing this word
            if word not in word_counter_dict:
              word_counter_dict[word] = {"spam": 0, "not_spam": 0}

            word_counter_dict[word]['spam' if is_spam else 'not_spam'] += 1

      return word_counter_dict

    def predict(self, datas, labels):
      # datas = datas[:2]
      # labels = labels[:2]

      correct = 0

      for i in range(len(datas)):
        text = datas[i]
        label = labels[i]
        print("Text:", text)

        spam_prob = math.log(self.positive_prior_prob)
        not_spam_prob = math.log(self.negative_prior_prob)
        words = text.split()

        for word in words:
            if(word in self.word_counter_dict):
                # Taking natural log of probabilities so it does not go to 0
                num_positive = self.word_counter_dict[word]['spam']
                not_num_positive = self.word_counter_dict[word]['not_spam']

                prob = (num_positive + A) / (num_positive + not_num_positive + A * D)
                spam_prob += math.log(prob)

                prob = (not_num_positive + A) / (num_positive + not_num_positive + A * D)
                not_spam_prob += math.log(prob)
            else:
                # Never seen the word in traning set
                pass

        print("Log sum of probabilities:", spam_prob, not_spam_prob)
        final_spam_prob = 1 - (spam_prob / (spam_prob + not_spam_prob))
        final_not_spam_prob = 1 - (not_spam_prob / (spam_prob + not_spam_prob))

        print("Label:", label)
        print("Spam probability", final_spam_prob)
        print("Not spam probability", final_not_spam_prob)

        is_spam = final_spam_prob > final_not_spam_prob
        print("Decision is spam:", is_spam)

        print("\n")

        if(is_spam and label == 1 or not is_spam and label == 0):
            correct += 1

      print("Accuracy", correct / len(labels))

    def fit(self, data, label):
        self.data = data
        self.label = label

        # Building a prio
        _, counter = np.unique(label, return_counts=True)
        self.num_positive = counter[1]
        self.num_negative = counter[0]

        self.positive_prior_prob = counter[1] / len(label)
        self.negative_prior_prob = counter[0] / len(label)

        print("Prior belief", self.positive_prior_prob, self.negative_prior_prob)

        # Creating the Bag of Words model
        self.word_counter_dict = self.build_vocabulary(data, label)
        self.num_of_spam_words = 0
        self.num_of_non_spam_words = 0

        for key in self.word_counter_dict:
          self.num_of_non_spam_words += self.word_counter_dict[key]['not_spam']
          self.num_of_spam_words += self.word_counter_dict[key]['spam']



def stemText(text):
  ps = PorterStemmer()
  z = []
  for i in text:
      z.append(ps.stem(i))
  w = z[:]
  z.clear()
  return w

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
