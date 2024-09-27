import joblib
import pandas as pd

model=joblib.load('model.pkl')

d=pd.read_csv('data-training\emotions.csv')

print(d['label'].head(5))

d=d.drop('label',axis=1)

y=model.predict(d.head(5))# negative=0, neutral=1, positive=2

print(y)