import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization

data = './weatherAUS.csv'
df = pd.read_csv(data)

#Exploring categorical data
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day
df.drop(['Date'], axis=1, inplace=True)

#Creating hot encodings
location = pd.get_dummies(df.Location, drop_first=True).head()
wind = pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
wind_9am = pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
wind_3pm = pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
rain_today = pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()

print(wind)

categorical = [var for var in df.columns if df[var].dtype=='O']
print('The categorical variables are :', categorical)
cat1 = [var for var in categorical if df[var].isnull().sum() != 0]

# Dropping unlabeled data
df.dropna(subset=['RainTomorrow'], inplace=True)

#Exploring numerical data
numerical = [var for var in df.columns if df[var].dtype!='O'] # Why O is representing non-numerical stuff??
print('The numerical variables are :', numerical)
print(df[numerical].isnull().sum())


# view summary statistics in numerical variables
for cat in numerical:
    print(df[cat].describe())

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


plt.figure(figsize=(15,10))


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



plt.show()






