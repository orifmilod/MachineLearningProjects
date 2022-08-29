import numpy as np
import pandas as pd
import queue
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets

class KNN:
    def __init__(self, K):
        self.K = K

    """ Input: Vector with (key, occurance_count) tuple """
    def get_most_freq(self, vector):
        most_freq = {}

        for item in vector:
            label = item[1]
            if label in most_freq:
                most_freq[label] += 1
            else:
                most_freq[label] = 1

        winner = None
        for key in most_freq:
            count = most_freq[key]
            if winner == None:
                winner = (key, count)
            elif winner[1] < count:
                winner = (key, count)

        return winner[0]

    """Fits the training data"""
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Defaults to Euclidean distance
    def minkowski_distance(self, a, b, p=2):
        distance = 0.0

        for i in range(len(a)):
            distance += abs(a[i] - b[i]) ** p

        return distance ** (1 / p)

    def predict(self, X_test, y_test):
        num_of_correct_classification = 0
        training_size = len(self.X_train)
        testing_size = len(X_test)

        for i in range(testing_size):
            q = queue.PriorityQueue()

            for j in range(training_size):
                distance = self.minkowski_distance(X_test[i], self.X_train[j])
                q.put((distance, self.y_train[j]))

            k_closest_neighbours = []
            for index in range(self.K):
                k_closest_neighbours.append(q.get())

            print("Closest neighbour", k_closest_neighbours)
            print("True label", y_test[i])
            most_freq = self.get_most_freq(k_closest_neighbours)
            print("Predicted", most_freq)

            if most_freq == y_test[i]:
                num_of_correct_classification += 1

        print(
            "Correct",
            num_of_correct_classification,
            num_of_correct_classification / testing_size,
        )


def main():
    mnist = datasets.load_digits()

    X_train, X_test, y_train, y_test = train_test_split(
        mnist.data, mnist.target, random_state=42
    )

    knn = KNN(4)
    knn.fit(X_train, y_train)
    knn.predict(X_test, y_test)

if __name__ == "__main__":
    main()
