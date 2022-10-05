import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#Load the data
df = pd.read_csv("wine_dataset.csv")

print(df.head())

# Select independent and dependent variable
y = df['target']
X =df.drop(['target'], axis=1)

# Split the dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))