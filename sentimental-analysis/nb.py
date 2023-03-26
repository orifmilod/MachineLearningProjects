import pandas as pd
import numpy as np
import string
from nltk.corpus import words
import nltk
import math
from nltk.stem.porter import PorterStemmer

# https://en.wikipedia.org/wiki/Additive_smoothing
A = 1
D = 2

def rm_punctuation_and_digits(text):
  try:
    return "".join([char for char in text if char not in string.punctuation + string.digits]).lower()
  except:
    print("Error:", text)


def remove_stopwords(word_vector):
    STOP_WORDS = nltk.corpus.stopwords.words("english")
    return [word for word in word_vector if word not in STOP_WORDS][1:]

def tokenize(text):
    return text.split()

def stemText(text):
  ps = PorterStemmer()
  z = []
  for i in text:
      z.append(ps.stem(i))
  w = z[:]
  z.clear()
  return w

class NaiveBayes:
    # Returns the probability of the words given that it"s spam or not spam
    def __init__(self, min_word_occurance):
      self.MIN_WORD_OCCURANCE = min_word_occurance
      self.word_set = set(words.words())

    def count_words(self, data, labels):
      if(len(data) != len(labels)):
        raise Exception("Inputs not same length")

      word_counter_dict = {}

      # Count number of words for both spam and not spam
      for i in range(len(data)):
        text = data[i]
        label = labels[i]
        is_negative = True if label == 0 else False

        print('text', type(text))
        words = text.split()
        for word in words:
            # Not an english word
            # if word not in self.word_set:
              # continue

            # First time seeing this word
            if word not in word_counter_dict:
              word_counter_dict[word] = {"positive": 0, "negative": 0}

            word_counter_dict[word]["negative" if is_negative else "positive"] += 1

      return word_counter_dict

    def predict(self, datas, labels):
      correct = 0

      for i in range(len(datas)):
        text = datas[i]
        label = labels[i]

        negative_prob = math.log(self.positive_prior_prob)
        positive_prob = math.log(self.negative_prior_prob)
        words = text.split()

        for word in words:
            if(word in self.word_counter_dict):
                # Taking natural log of probabilities so it does not go to 0
                num_positive = self.word_counter_dict[word]["negative"]
                not_num_positive = self.word_counter_dict[word]["positive"]

                prob = (num_positive + A) / (num_positive + not_num_positive + A * D)
                negative_prob += math.log(prob)

                prob = (not_num_positive + A) / (num_positive + not_num_positive + A * D)
                positive_prob += math.log(prob)
            else:
                # Never seen the word in traning set
                pass

        # print("Log sum of probabilities:", negative_prob, positive_prob)
        final_negative_prob = 1 - (negative_prob / (negative_prob + positive_prob))

        print("Label:", label)
        print("Spam probability", final_negative_prob)

        is_negative = final_negative_prob > 0.55 # final_not_spam_prob
        print("Decision is negative:", is_negative)

        print("\n")

        if(is_negative and label == 0 or not is_negative and label == 1):
            correct += 1

      print("Accuracy", correct / len(labels))

    def train(self, data, label):
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
        self.word_counter_dict = self.count_words(data, label)

def preprocess(data):
  data.dropna(inplace=True)
  data.drop(data[data.feedback == "Irrelevant"].index, inplace=True)
  data.drop(data[data.feedback == "Neutral"].index, inplace=True)

  # ["Positive" "Neutral" "Negative" "Irrelevant"]
  # Something diff for Irrelevant?
  data.feedback.replace(["Positive", "Negative"], [1, 0], inplace=True)

  # Data cleaning
  data["text"] = data["text"].apply(lambda text:rm_punctuation_and_digits(text))
  data["text"] = data["text"].apply(lambda text:tokenize(text))
  data["text"] = data["text"].apply(lambda text:remove_stopwords(text))
  # data["text"] = data["text"].apply(stemText) - why does stemming hurt the accuracy?
  data["text"] = data["text"].apply(lambda text: " ".join(text))
  data.drop_duplicates(subset=['text'])
  data.drop(data[data.text == ""].index, inplace=True)
  return data

def main():
    training_data = pd.read_csv("./data/twitter_training.csv")
    test_data = pd.read_csv("./data/twitter_validation.csv")

    # TODO: Use company overall feedback probability
    training_data.columns = ["idk", "about", "feedback", "text"]
    test_data.columns = ["idk", "about", "feedback", "text"]

    training_data = preprocess(training_data)
    test_data = preprocess(test_data)

    nb = NaiveBayes(3)
    nb.train(training_data.text.to_numpy(), training_data.feedback.to_numpy())
    nb.predict(test_data.text.to_numpy(), test_data.feedback.to_numpy())

if __name__ == "__main__":
    main()

