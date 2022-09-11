import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization


def clean(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe['Day'] = dataframe['Date'].dt.day
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe.drop(['Date'], axis=1, inplace=True)
    return dataframe


def main():
    data = './weatherAUS.csv'
    df = pd.read_csv(data)

    #Exploring categorical data
    df = clean(df)

    #Creating one-hot encodings for categorical data
    location = pd.get_dummies(df.Location, drop_first=True).head()
    wind = pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
    wind_9am = pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
    wind_3pm = pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
    rain_today = pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()

    categorical = [var for var in df.columns if df[var].dtype=='O']
    print('The categorical variables are :', categorical)
    categories_wo_labale = [var for var in categorical if df[var].isnull().sum() != 0]

    # Dropping unlabeled data
    # TODO: what is alternative way of handlign missing labels?
    for category in categories_wo_labale:
        df.dropna(subset=[category], inplace=True)


    #Exploring numerical data
    numerical = [var for var in df.columns if df[var].dtype != 'O']
    print('The numerical variables are :', numerical)
    # TODO: Clean up missing variables
    print(df[numerical].isnull().sum())

    # view summary statistics in numerical variables
    print(df[numerical].describe())

    plt.figure(figsize=(15,10))


    plt.subplot(2, 2, 1)
    fig = df.boxplot(column='Rainfall')
    fig.set_title('')
    fig.set_ylabel('Rainfall')

    plt.subplot(2, 2, 2)
    fig = df.boxplot(column='Evaporation')
    fig.set_title('')
    fig.set_ylabel('Evaporation')


    plt.subplot(2, 2, 3)
    fig = df.boxplot(column='WindSpeed9am')
    fig.set_title('')
    fig.set_ylabel('WindSpeed9am')


    plt.subplot(2, 2, 4)
    fig = df.boxplot(column='WindSpeed3pm')
    fig.set_title('')
    fig.set_ylabel('WindSpeed3pm')


    plt.subplot(2, 2, 1)
    fig = df.Rainfall.hist(bins=10)
    fig.set_xlabel('Rainfall')
    fig.set_ylabel('RainTomorrow')


    plt.subplot(2, 2, 2)
    fig = df.Evaporation.hist(bins=10)
    fig.set_xlabel('Evaporation')
    fig.set_ylabel('RainTomorrow')


    plt.subplot(2, 2, 3)
    fig = df.WindSpeed9am.hist(bins=10)
    fig.set_xlabel('WindSpeed9am')
    fig.set_ylabel('RainTomorrow')


    plt.subplot(2, 2, 4)
    fig = df.WindSpeed3pm.hist(bins=10)
    fig.set_xlabel('WindSpeed3pm')
    fig.set_ylabel('RainTomorrow')

    plt.figure(figsize=(15,10))
    plt.show()

if __name__ == '__main__':
    main()
