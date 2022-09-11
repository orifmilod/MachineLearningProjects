import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

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

    categorical = [var for var in df.columns if df[var].dtype=='O']
    for col in categorical:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

def feature_scale(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

def main():
    data = './weatherAUS.csv'
    df = pd.read_csv(data)

    # Exploring categorical data
    df = clean(df)

    categorical = [var for var in df.columns if df[var].dtype=='O']
    # print('The categorical variables are :', categorical)
    categories_wo_labale = [var for var in categorical if df[var].isnull().sum() != 0]

    # Dropping unlabeled data
    # TODO: what is alternative way of handlign missing labels?
    for category in categories_wo_labale:
        df.dropna(subset=[category], inplace=True)


    #Exploring numerical data
    numerical = [var for var in df.columns if df[var].dtype != 'O']
    # print('The numerical variables are :', numerical)
    # TODO: Clean up missing variables

    # View summary statistics in numerical variables
    # print(df[numerical].describe())

    # Feature engineering
    df = fill_na(df)
    df = top_code(df, numerical)

    # Declare feature vector and target variable
    X = df.drop(['RainTomorrow'], axis=1)
    y = df['RainTomorrow']
    categorical.remove('RainTomorrow')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Binary encodings for categorical data
    encoder = ce.BinaryEncoder(cols=categorical)
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)
 
    # TODO: Check Dummy encoding vs Binary encoding performance difference
    # X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     # pd.get_dummies(X_train.Location), 
                     # pd.get_dummies(X_train.WindGustDir),
                     # pd.get_dummies(X_train.WindDir9am),
                     # pd.get_dummies(X_train.WindDir3pm)], axis=1)

    feature_scale(X_train, X_test)
    print(X_train.describe())

    # plot_data(df)

if __name__ == '__main__':
    main()
