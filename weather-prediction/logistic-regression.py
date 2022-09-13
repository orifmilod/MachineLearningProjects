import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
import math
from random import random

def plot_data(dataframe):
    # Draw boxplots to visualize outliers
    plt.figure(figsize=(15,10))

    plt.subplot(2, 2, 1)
    fig = dataframe.boxplot(column='Rainfall')
    fig.set_title('')
    fig.set_ylabel('Rainfall')


    plt.subplot(2, 2, 2)
    fig = dataframe.boxplot(column='Evaporation')
    fig.set_title('')
    fig.set_ylabel('Evaporation')

    plt.subplot(2, 2, 3)
    fig = dataframe.boxplot(column='WindSpeed9am')
    fig.set_title('')
    fig.set_ylabel('WindSpeed9am')

    plt.subplot(2, 2, 4)
    fig = dataframe.boxplot(column='WindSpeed3pm')
    fig.set_title('')
    fig.set_ylabel('WindSpeed3pm')

    # Plot histogram to check distribution
    plt.figure(figsize=(15,10))

    plt.subplot(2, 2, 1)
    fig = dataframe.Rainfall.hist(bins=10)
    fig.set_xlabel('Rainfall')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 2)
    fig = dataframe.Evaporation.hist(bins=10)
    fig.set_xlabel('Evaporation')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 3)
    fig = dataframe.WindSpeed9am.hist(bins=10)
    fig.set_xlabel('WindSpeed9am')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 4)
    fig = dataframe.WindSpeed3pm.hist(bins=10)
    fig.set_xlabel('WindSpeed3pm')
    fig.set_ylabel('RainTomorrow')
    plt.show()

def clean(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe['Day'] = dataframe['Date'].dt.day
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe.drop(['Date'], axis=1, inplace=True)
    return dataframe


# Top-coding numerical outliers
def top_code(df, categories):
    for cat in categories:
        IQR = df[cat].quantile(0.75) - df.Rainfall.quantile(0.25)
        lower_bound = df[cat].quantile(0.25) - (IQR * 3)
        top_bound = df[cat].quantile(0.75) + (IQR * 3)
        df[cat] = np.where(df[cat] > top_bound, top_bound, df[cat])
        # print('"{cat}" outliers are values < {lowerboundary} or > {upperboundary}'.format(cat=cat, lowerboundary=lower_bound, upperboundary=top_bound))
    return df

def fill_na(df):
    # Impute missing variables with most frequent value
    numerical = [var for var in df.columns if df[var].dtype != 'O']
    for col in numerical:
        col_median = df[col].median()
        df[col].fillna(col_median, inplace=True)

    categorical = [var for var in df.columns if df[var].dtype == 'O']
    for col in categorical:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def feature_scale(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


class LogisticRegression:
    epochs = 100
    learning_rate = 0.05

    # Negative-loss likelihood is loss/cost function for Logisitc regression
    def loss(self, label, prediction):
        # TODO: Take care of 0/1 predictions
        return -((label * math.log(prediction)) + ((1 - label) * math.log(1 - prediction)))

    def error(self, labels, predictions):
        # assert len(labels) == len(predictions)
        num_items = len(labels)
        sum_of_errors = sum([self.loss(y, y_pred) for y, y_pred in zip(labels, predictions)])
        return (1 / num_items) * sum_of_errors

    def sigmoid(self, x):
        return (1 / (1 + math.exp(-x)))

    def squish(self, beta, x):
        # assert len(beta) == len(x)
        # Calculate the dot product and pass it in sigmoid function
        return self.sigmoid(np.dot(beta, x))

    def __init__(self, ):
        self.beta = []

    def fit(self, data, labels):
        assert len(data) == len(labels)
        self.beta = [random() / 10 for _ in range(len(data[0]))]

        for epoch in range(self.epochs):
            predicted_labels = [self.squish(self.beta, x) for x in data]
            loss = self.error(labels, predicted_labels)
            print(f'Epoch {epoch} --> loss: {loss}')

            # Calculating gradient
            grad = [0 for _ in range(len(self.beta))]

            for x, y in zip(data, labels):
                err = self.squish(self.beta, x) - y
                for i, x_i in enumerate(x):
                    grad[i] += (err * x_i)

                grad = [1 / len(x) * g_i for g_i in grad]

            # Take a small step in the direction of greatest decrease
            new_beta = [b - (gb * self.learning_rate) for b, gb in zip(self.beta, grad)]
            self.beta = new_beta


    def predict(self, data, labels):
        predicted_labels = [self.squish(self.beta, x) for x in data]
        # print("Prediciton", predicted_labels)
        counter = 0
        for i in range(len(predicted_labels)):
            if(predicted_labels[i] >= 0.5 and labels[i] == 1):
                counter += 1
            elif(predicted_labels[i] < 0.5 and labels[i] == 0):
                counter += 1
        
        print("Accuracy: {counter}".format(counter = counter/len(labels)))


def main():
    data = './weatherAUS.csv'
    df = pd.read_csv(data)

    # Exploring categorical data
    df = clean(df)

    categorical = [var for var in df.columns if df[var].dtype == 'O']
    categories_wo_labale = [var for var in categorical if df[var].isnull().sum() != 0]

    # Dropping unlabeled data
    # TODO: what is alternative way of handling missing labels?
    for category in categories_wo_labale:
        df.dropna(subset=[category], inplace=True)

    #Exploring numerical data
    numerical = [var for var in df.columns if df[var].dtype != 'O']

    # Feature engineering
    df = fill_na(df)
    df = top_code(df, numerical)

    # Declare feature vector and target variable
    df['RainTomorrow'] = df['RainTomorrow'].eq('Yes').mul(1)

    X = df.drop(['RainTomorrow'], axis=1)
    y = df['RainTomorrow']
    categorical.remove('RainTomorrow')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Binary encodings for categorical data
    encoder = ce.BinaryEncoder(cols=categorical)
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
 
    # TODO: Check Dummy encoding vs Binary encoding performance/practical difference
    # X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     # pd.get_dummies(X_train.Location), 
                     # pd.get_dummies(X_train.WindGustDir),
                     # pd.get_dummies(X_train.WindDir9am),
                     # pd.get_dummies(X_train.WindDir3pm)], axis=1)


    X_train, X_test = feature_scale(X_train, X_test)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr.predict(X_test, y_test.to_numpy())

    # Sk-learn LR 
    # logreg = LR(solver='liblinear', random_state=0)
    # logreg.fit(X_train, y_train)
    # y_pred_test = logreg.predict(X_test)

    # print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

    # plot_data(df)

if __name__ == '__main__':
    main()
