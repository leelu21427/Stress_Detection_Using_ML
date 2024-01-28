# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import numpy as np
import seaborn as sb

df=pd.read_csv("Stress Data_C.csv")

X_train, X_test, y_train, y_test=train_test_split(df.iloc[:, :8], df['stress_level'],test_size=0.2, random_state=8)
X = df[['snoring_rate', 'body_temperature','blood_oxygen','respiration_rate', 'sleeping_hours', 'heart_rate','Headache','Working_hours']]
y = df['stress_level']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest_classifier.fit(X_train, y_train)
random_forest_classifier.predict(X_test)




pickle.dump(random_forest_classifier,open("model.pkl","wb"))
# -*- coding: utf-8 -*-
