import numpy as np
import pandas as pd
import queue
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets


class KNN:
    def __init__(self, K):
        self.K = K

    """ Input: Queue with (key, occurance_count) tuple """
    def get_most_freq(queue):
        most_freq = {}

        while not queue.empty():
            if label in most_freq:
                most_freq[label] += 1
            else:
                most_freq[label] = 1

        winner = None
        for key in most_freq:
            count = most_freq[key]
            if (winner ==  None):
                winner = (key, count)
            elif(winner[1] < count):
                winner = (key, count)

        return winner[0]

    """Fits the training data"""
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.Y_train = y_train
        num_of_correct_classifcation = 0

        n = len(X_train)
        for i in range(n):
            q = queue.PriorityQueue()
            for j in range(n):
                if i == j:
                    continue
                dist = self.minkowski_distance(X_train[i], X_train[j])
                q.put((dist, y_train[j]))

            k_closest_neighbours = []
            for i in range(self.K):
                k_closest_neighbours.append(.get())

            most_freq = self.get_most_freq(k_closest_neighbours)
            if(most_freq == y_train[i]):
                num_of_correct_classifcation +=1
                
        print("Correct", num_of_correct_classifcation, num_of_correct_classifcation/len(X_train))

    def minkowski_distance(self, a, b, p=2):
        distance = 0.0

        for i in range(len(a)):
            distance += abs(a[i] - b[i]) ** p

        return  distance ** (1/p)

    def predict(self, X_tests, y_tests):
        num_of_correct_classifcation = 0
        for i in range(len(self.X_train)):
            q = queue.PriorityQueue(self.K)

            for j in range(len(self.X_train)):
                if i == j:
                    continue
                dist = self.minkowski_distance(self.X_train[i], self.X_train[j])
                q.put((dist, self.y_train[j]))

            most_freq = {}

            while (q.empty() == False):
                label = q.get()[1]
                if(most_freq[label]):
                    most_freq[label] += 1
                else:
                    most_freq[label] += 0


def __init__():
  mnist = datasets.load_digits()

# print(mnist.data[1])
# print(mnist.target[1])
# plt.matshow(mnist.images[1])
# plt.show()

  X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, random_state=0)
  X_train = X_train[:900]
  y_train = y_train[:900]
  knn = KNN(4)
  knn.fit(X_train, y_train)
# print(knn.predict(X_test))
