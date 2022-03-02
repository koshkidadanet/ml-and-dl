import numpy as np
import pandas as pd
import pandas
import tensorflow as tf
import csv
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

dataSurv = pd.read_csv('gender_submission.csv')
dataTrain = pd.read_csv('train.csv')
dataTest = pd.read_csv('test.csv')

dataTrain.dropna(subset=["Pclass", "Sex", "SibSp", "Parch",'Survived'], inplace=True) #Удаляем пустые данные
dataTrain['Sex'] = dataTrain['Sex'].replace(to_replace=['male', 'female'], value=[1, 0])
#dataTrain['Survived'] = dataTrain['Survived'].replace(to_replace=[1, 0], value=[, 0])

dataTest['Survived'] = dataSurv['Survived']
dataTest.dropna(subset=["Pclass", "Sex", "SibSp", "Parch",'Survived' ], inplace=True) #Удаляем пустые данные
dataTest['Sex'] = dataTest['Sex'].replace(to_replace=['male', 'female'], value=[1, 0])
#dataTest['Survived'] = dataTest['Survived'].replace(to_replace=[1, 0], value=[50, 0])

train_x = dataTrain[["Pclass", "Sex", "SibSp", "Parch"]].to_numpy()
train_y = dataTrain[['Survived']].to_numpy()

test_x = dataTest[["Pclass", "Sex", "SibSp", "Parch"]].to_numpy()
test_y = dataTest[['Survived']].to_numpy()

encoder = OneHotEncoder(sparse=False)
#train_y = encoder.fit_transform(train_y)
#est_y = encoder.transform(test_y)


#Создаём нейронную сеть
#,kernel_regularizer='l2_l1'
#categorical_crossentropy
#binary_crossentropy
#RMSprop()
model = keras.Sequential([
                     keras.layers.Dense(16, input_shape=(4,), activation='relu'),
                     keras.layers.Dense(16, activation='relu'),
                     keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy',metrics= ['BinaryAccuracy'])
print(model.summary())

model.fit(train_x, train_y, verbose=2,batch_size=3, epochs=200)

test_loss, test_acc = model.evaluate(train_x, train_y)
print('Тренировочная точность = ', test_acc)

test_loss, test_acc = model.evaluate(test_x, test_y)
print('Тестовая точность = ', test_acc)

history_dict = model.history
#print(history_dict)

prediction = model.predict(test_x)
number = 11
#print(prediction)
"""
for i in range(len(test_x)):
  print('Предсказываю (округленные) = ',round(prediction[i][0]),
        #'\n','Предсказываю  = ',prediction[i],
        '\n','Ответ = ',test_y[i],sep='')
"""


prediction = model.predict(test_x)
prediction = pd.DataFrame(prediction)
tit = dataTest.iloc[: ,:1]
tit['Survived'] = prediction[0].round()
tit.set_index('PassengerId', inplace=True)
tit = tit.astype({'Survived': np.int})
print(tit)
tit.to_csv('Titanic.csv', sep=',')


clf = DecisionTreeClassifier(random_state=1)
clf.fit(train_x, train_y)

prediction = clf.predict(train_x)
score = r2_score(train_y ,prediction)
print(score)

prediction = clf.predict(test_x)
prediction = pd.DataFrame(prediction)

#print(prediction2)
tit = dataTest.iloc[: , :1]
tit['Survived'] = prediction
tit.set_index('PassengerId', inplace=True)
tit = tit.astype({'Survived': np.int})
#print(tit)
tit.to_csv('Titanic.csv', sep=',')


clf = RandomForestClassifier(n_estimators=100,random_state=1)
clf.fit(train_x, train_y)

prediction = pd.DataFrame(clf.predict(test_x))
#print(prediction2)
tit = dataTest.iloc[: , :1]
tit['Survived'] = prediction
tit.set_index('PassengerId', inplace=True)
tit = tit.astype({'Survived': np.int})
#print(tit)
tit.to_csv('Titanic.csv', sep=',')
