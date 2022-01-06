import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import imblearn  # used for imbalanced dataset 
from imblearn.under_sampling import RandomUnderSampler 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense

from sklearn import metrics
from sklearn.metrics import confusion_matrix # To make the confusion matrix
from sklearn.metrics import roc_curve # To make the receiver operating characteristic curve(only for binary classification task)
from sklearn.metrics import roc_auc_score # To find the area of the receiver operating characteristic curve from prediction scores
from sklearn.metrics import auc # To find the area of the receiver operating characteristic curve
from sklearn.metrics import precision_recall_curve # To find precision-recall
credit_record = pd.read_csv("creditcard.csv") # This is used to read the data
print(credit_record.head(8)) # This helps us select the first 8 entry of our data
credit_record = credit_record.drop("Time", axis=1) #To remove/drop the Time field from the current 31 fields as it is not needed
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()  # Used to standardize the function by removing the mean and scaling to the unit variance
credit_record['std_Amount'] = scaler.fit_transform(credit_record['Amount'].values.reshape (-1,1)) #To add the std_Amount field from the current 30 fields and value of it should be standard scale of Amount to -1 to 1
credit_record = credit_record.drop("Amount", axis=1) # To remove/drop the Amount field from the current 31 fields as it is not needed because now we have std_Amount
print(credit_record.head(8)) # To print and check the credit_record of the first 8 entry of our data
print(sns.countplot(x="Class", data=credit_record)) #here we are plotting the graph X=Class of the above data and Y access is the amount of data(in this case 284806)
plt.show()
undersample = RandomUnderSampler(sampling_strategy=0.5) # Class to perform random under-sampling, we can under sample the majority class/classes by randomly picking some of the samples with or without replacement.
cols = credit_record.columns.tolist() # take all fields(we can say it as a coulumn) in a list form rather than object type
cols = [c for c in cols if c not in ["Class"]] # if C is not class then it will be included in the cols
target = "Class" # setting target as Class
print(cols) #printing column to check it
X = credit_record[cols] # defining X of graph which is equal to field from dataset credit_record(credit_record[cols])
Y = credit_record[target] # defining Y of graph which is equal to Class from dataset credit_record(credit_record[target])
X_under,Y_under = undersample.fit_resample(X,Y) # setting X_under,Y_under by resampling the data as per X and Y
from pandas import DataFrame
test = pd.DataFrame(Y_under, columns = ['Class']) # visualizing undersampling results
fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
print(sns.countplot(x="Class", data=credit_record, ax=axs[0]))
plt.show()
print(sns.countplot(x="Class", data=test, ax=axs[1]))
plt.show()
fig.suptitle("Class repartition before and after undersampling")
a1=fig.axes[0] 
a1.set_title("Before")
a2=fig.axes[1] 
a2.set_title("After")
model = Sequential() # Sequential is used as the layer because we are making the model layer by layer
model.add(Dense(32, input_shape=(29,), activation='relu')), #used for implementing densely-connected neural network layer
model.add(Dropout(0.2)), #prevent overfitting by setting input units=0 with a frequency of 0.2
model.add(Dense(16, activation='relu')),
model.add(Dropout(0.2)),
model.add(Dense(8, activation='relu')),
model.add(Dropout(0.2)),
model.add(Dense(4, activation='relu')),
model.add(Dropout(0.2)),
model.add(Dense(1, activation='sigmoid'))
# Different type of activation functions : RRelu (Best but not available for tensorflow so go for next best) Relu Sigmoid Softmax(to bring the value in the range of 0 to 1) Tanh
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001) #optimizer is used to update weights and learning rate, other optimizer Adam(Best) Adagrad SpareAdam Stocastic Gradient Descent
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy']) #metrics loss function binary_cross_entropy(Mostly used and it is usually used because it used for two values usually 0 and 1 so we can use SparseCategoricalCrossentropy for more than two) cross_entropy CategoricalCrossentropy(have to make sure it is in vector form) mse_loss
earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=15, verbose=1,mode='auto', baseline=None, restore_best_weights=False) #Used to stop training the model when metric stops improving
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=1) # to split the data into two parts one is for trainging and one for testing
history = model.fit(X_train.values, y_train.values, epochs = 6, batch_size=5, validation_split = 0.15, verbose = 1,callbacks = [earlystopper]) #for making the data in tuple(X_train.values, y_train.values) 
# epochs is the number of times the model will be trained
# validation_split is used to reserve part of your training data for validation
# batch_size the batch size from the dataset
# callback is what type of call back will we be using here we are using earlystopper as defined above, we can also use History, BaseLogger, etc
history_dict = history.history # to make the record in the history object
# trying to plot a graph of training loss and validation training loss
loss_values = history_dict['loss'] #taking training loss value from the trained data
val_loss_values=history_dict['val_loss'] #taking validation loss value from the trained data
plt.plot(loss_values,'b',label='training loss') #taking the above loss value and giving it a color blue and labeling it training loss
plt.plot(val_loss_values,'r',label='val training loss') #taking the above validation loss value and giving it a color red and labeling it val training loss
plt.legend() #Legend of the graph
plt.xlabel("Epochs") # no of epochs
plt.show()
# trying to plot a graph of training accuracy and validation accuracy
accuracy_values = history_dict['accuracy'] #taking training accuracy value from the trained data
val_accuracy_values=history_dict['val_accuracy'] #taking validation accuracy value from the trained data
plt.plot(val_accuracy_values,'-r',label='val_accuracy') #taking the above val_accuracy value and giving it a color red and making it a solid line and labeling it val_accuracy
plt.plot(accuracy_values,'-b',label='accuracy') #taking the above accuracy value and giving it a color blue and making it a solid line and labeling it accuracy
plt.legend() # legend of the graph
plt.xlabel("Epochs") # no of epochs
plt.show()
y_pred_nn = model.predict_classes(X_test) # make predicitions using the above model
print("Accuracy Neural Net:",metrics.accuracy_score(y_test, y_pred_nn)) # Accuracy by taking the test data and predicted data of neural network
print("Precision Neural Net:",metrics.precision_score(y_test, y_pred_nn)) # Accuracy by taking the test data and predicted data of neural network
print("Recall Neural Net:",metrics.recall_score(y_test, y_pred_nn)) # Accuracy by taking the test data and predicted data of neural network
print("F1 Score Neural Net:",metrics.f1_score(y_test, y_pred_nn)) # Accuracy by taking the test data and predicted data of neural network
matrix_nn = confusion_matrix(y_test, y_pred_nn) # make the confusion matrix using the y_test and y_pred_nn
cm_nn = pd.DataFrame(matrix_nn, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud']) # make the data into a 2d table of rows and columns
sns.heatmap(cm_nn, annot=True, cbar=None, cmap="YlGnBu", fmt = 'g') # make the confusion matrix using the y_test and y_pred_nn
plt.title("Confusion Matrix Neural Network"), plt.tight_layout() # make the confusion matrix using the y_test and y_pred_nn
plt.ylabel("True Class"), plt.xlabel("Predicted Class") # make the confusion matrix using the y_test and y_pred_nn
plt.show() # make the confusion matrix using the y_test and y_pred_nn
y_pred_nn_proba = model.predict(X_test) # make predicitions using the above model
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_pred_nn_proba) # make roc curve from above
auc_keras = auc(fpr_keras, tpr_keras)
print('AUC Neural Net: ', auc_keras)# printing the AUC Neural Net
plt.figure(1) # to create a new figure
plt.plot([0, 1], [0, 1], 'k--') #plot the value of the above 
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras)) #plot the value of the above fpr_keras, tpr_keras with label 
plt.xlabel('False positive rate') #setting x label as False positive rate
plt.ylabel('True positive rate') #setting y label as True positive rate
plt.title('Neural Net ROC curve') # Set the tile as Neural Net ROC curve
plt.legend(loc='best')# legend of the graph and the location is the best
plt.show()# plot the above data
nn_precision, nn_recall, _ = precision_recall_curve(y_test, y_pred_nn_proba.reshape(-1, 1)[0]) # for computing the precision-recall pairs of y_test and y_pred_nn_proba(needs to be reshaped to same as y_test)
no_skill = len(y_test[y_test==1]) / len(y_test) # setting no_skill value len(y_test[y_test==1])/len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill') # plot data of x will be[0, 1] and y will be [no_skill, no_skill] with dashed line and color os black with label as No Skill
plt.plot(nn_recall, nn_precision, color='orange', label='TF NN') # plot data of nn_recall and nn_precision with orange line and label is TF NN
plt.xlabel('Recall') # setting x label as Recall
plt.ylabel('Precision') # setting y label as Precision
plt.title('Precision-Recall curve') # Set the tile as Precision-Recall curve
plt.legend()# legend of the graph
plt.show()# plot the above data
