import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

data=pd.read_csv('data-training/emotions.csv')

x=data.drop('label',axis=1)
y=data['label']

y=y.replace('NEGATIVE',0)
y=y.replace('NEUTRAL',1)
y=y.replace('POSITIVE',2)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)

#decision tree

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)

#random forest

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
pred_rf=rf.predict(x_test)

print('Accuracy of Decision Tree:\t\t',accuracy_score(y_pred_dt,y_test)*100)
print('Accuracy of Random Forest:\t\t',accuracy_score(pred_rf,y_test)*100)


joblib.dump(rf, 'model.pkl')