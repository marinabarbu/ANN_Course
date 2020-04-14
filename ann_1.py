#ANN
#install Theano
#install TensorFlow
#install keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y= dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) #codare pentru franta - 0, spania, germania
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2]) #codare male/female
ct = ColumnTransformer([("score", OneHotEncoder(),[1])],remainder='passthrough')
#onehotencoder = OneHotEncoder([1])
#onehotencoder = OneHotEncoder(categorical_features = [1])
X = ct.fit_transform(X)
X = X[:,1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
#scalare date:
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#module keras:
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
# step 1 initializare ponderi cu valori mici prin functia Dense
#step 2 input layer va fi de dim 11
# step 3: fct de activare: sigmoida pt output layer si rectified pt hidden layeers
#step 4: comparare rezultate - calc eroare
# step5: back propag -> update la ponderi
#step6: repetare pasi 1-5 si update la ponderi dupa fiecare obs/sau mai multe obs
#se repeta pt fiecare epoca si se calc loss


#stratul de intrare + primul strat ascuns:
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim=11))

#al 2-lea strat ascuns

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#stratul de iesire:
    
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#compiling the ANN

classifier.compile(optimizer = 'adam',  loss='binary_crossentropy', metrics = ['accuracy'])

#fitting to the training set

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

y_pred = classifier.predict(X_test)
y_pred =(y_pred > 0.5)


new_prediction=classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction >0.5)
print("New pred: ", new_prediction)
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test, y_pred)

#X_homework = 
