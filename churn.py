<------------------------------------Churn Prediction Using Keras--------------------->>>>>>>>>

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore') 

#Importing the Dataset
dataset=pd.read_csv('C:/Users/nsidd/OneDrive/Desktop/Churn_Modelling.csv')
dataset.head()

#Splitting into X,Y
X=dataset.iloc[:,3:13].values
#X
y=dataset.iloc[:,13].values

#Importing Label Encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) #Encoding the values of column Country
labelencoder_X_2=LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
#X

#Splitting into Train and Test
from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 
#X_train

from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
#X_train

#Importing Keras
import keras 
from keras.models import Sequential 
from keras.layers import Dense 

classifier=Sequential()

#Passing Number of input and output values with Relu Activation
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#As the output is a binary we use'binary_crossentropy' and Accuracy for Metrics 
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#training the data with 100 Epochs
classifier.fit(X_train, y_train,batch_size=10,nb_epoch=100)

#Predicting the test data
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

#Confusion Matrix Output----->>>>> [[1552,   43],
                                    [ 274,  131]]

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

#Accuracy Obtained---------------->>>>>>> 84.15%