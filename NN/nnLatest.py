#Libraries
import numpy as np
import pandas as pd
import random as r
import matplotlib.pyplot as plot

from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

 #@author Wei Jun Shong
 #@author Seow Ke Ni

#pd.set_option("display.max_rows", None, "display.max_columns", None) #to print everything


#read dataset
dataSet= pd.read_csv('dataset_37_diabetes.csv')
dataSet.head()
print(dataSet)
print()


#Obtain data
X = dataSet.iloc[:,1:-1].values
Y = dataSet.iloc[:,-1].values


#Split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=1 )#X independent,Y dependent, (80% trainning, 20% testing)
#X_train = X_train/np.max(X_train,axis=0) #line above #random_state=1

print("80% trainning for X(independent variables)")
print(X_train)#80% trainning for X(independent variables)
print()

print("20% testing for X(independent variables)")
print(X_test)#20% testing for X(independent variables)
print()

print("80% trainning for Y(dependent variables/target)")
print(Y_train)#80% trainning for Y(dependent variables/target)
print()

print("20% testing for Y(dependent variables/target)")
print(Y_test)#20% testing for Y(dependent variables/target)
print()

#Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Bias and Threshold
print("The bias and threshold is")
def learn(input_vector,weights,bias):
    layer_1 = np.dot(input_vector,weights)+bias #hidden layer
    layer_2 = sigmoid(layer_1) #output layer using binary sigmoid
    return layer_2 #returning hidden neurons

nrows, ncolumns = X_train.shape[0], X_train.shape[1]
weights = [[r.random() for i in range(nrows)] for j in range(ncolumns)]#creates random decimals from 0 to 1
weights = np.array(weights)
v=learn(X_train, weights,1)

print(v)

#Data balancing
print('\n--- Class balance ---')
print(np.unique(Y_train, return_counts=True))
print(np.unique(Y_test, return_counts=True))

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=150, alpha=1e-4,solver='adam', verbose=0, tol=1e-8, random_state=1,learning_rate_init=.01, activation='logistic')

N_TRAIN_SAMPLES = X_train.shape[0]
N_TEST_SAMPLES = X_test.shape[0]
N_EPOCHS = 150
N_BATCH = 128
N_CLASSES1 = np.unique(Y_train)
N_CLASSES2 = np.unique(Y_test)

scores_train = []
scores_test = []

#EPOCH
epoch = 0
while epoch < N_EPOCHS:
    print('epoch: ', epoch)
    #SHUFFLE
    r_per= np.random.permutation(X_train.shape[0])
    mini_batch_index = 0
    while True:
        #Index
        indicator = r_per[mini_batch_index:mini_batch_index + N_BATCH]
        mlp.partial_fit(X_train[indicator], Y_train[indicator], classes=N_CLASSES1)
        mini_batch_index += N_BATCH
        if mini_batch_index >= N_TRAIN_SAMPLES:
          break
        #Trainning score
        scores_train.append(mlp.score(X_train, Y_train))
        epoch += 1

# EPOCH
epoch = 0
while epoch < N_EPOCHS:
    print('epoch: ', epoch)
    # SHUFFLING
    r_per = np.random.permutation(X_test.shape[0])
    mini_batch_index = 0
    while True:
        #Index
        indicator = r_per[mini_batch_index:mini_batch_index + N_BATCH]
        mlp.partial_fit(X_test[indicator], Y_test[indicator], classes=N_CLASSES2)
        mini_batch_index += N_BATCH
        if mini_batch_index >= N_TEST_SAMPLES:
          break
        #Testing score
        scores_test.append(mlp.score(X_test, Y_test))
        epoch += 1

#Training accuracy graph
def trainAccGraph():
   plot.plot(scores_train, 'b')
   plot.title('Model Accuracy')
   plot.xlabel('Epoch')
   plot.ylabel('Accuracy')
   plot.legend(['train'], loc='upper left')
   plot.show()

#Testing accuracy graph
def testAccGraph():
   plot.plot(scores_test, 'r')
   plot.title('Model Accuracy')
   plot.xlabel('Epoch')
   plot.ylabel('Accuracy')
   plot.legend(['test'], loc='upper left')
   plot.show()

#MSE for training
def TRAIN_MSE():
    mlp.fit(X_train, Y_train)
    plot.plot(mlp.loss_curve_, 'b')
    plot.title('Model Mse')
    plot.xlabel('Epoch')
    plot.ylabel('MSE')
    plot.legend(['training'], loc='upper left')
    plot.show()

#MSE for testing
def TEST_MSE():
    mlp.fit(X_test, Y_test)
    plot.plot(mlp.loss_curve_, 'r')
    plot.title('Model Mse')
    plot.xlabel('Epoch')
    plot.ylabel('MSE')
    plot.legend(['test'], loc='upper left')
    plot.show()

#Frame
root = Tk()
root.title("Classification Of Diabetes Patient Using ANN")
root.geometry("420x230")
root.configure(bg='blue')
root.eval('tk::PlaceWindow . center')
root.resizable(False, False)

#Training accuracy graph
redbutton = Button(root, text='Training accuracy',
                   command=trainAccGraph, fg='blue',
                   height=2, width=35)
redbutton.place(x=88, y=20)

#Testing accuracy graph
bluebutton= Button(root, text='Testing accuracy',
                   command=testAccGraph, fg='red',
                   height=2, width=35)
bluebutton.place(x=88, y=70)

#Training MSE graph
brownbutton  = Button(root, text='Training MSE',
                    command=TRAIN_MSE,fg='brown',
                    height=2, width=35)
brownbutton .place(x=88, y=120)

#Testing MSE graph
blackbutton = Button(root, text='Testing MSE',
                    command=TEST_MSE, fg='black',
                    height=2, width=35)
blackbutton.place(x=88, y=170)
root.mainloop()
