import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
import random

def xgBoostKfold(k):
  trainData, testData, dtypes = load_csv()
  xTrain, yTrain, xTest, yTest = pre_processing(trainData, testData, dtypes)
  kfold = KFold(n_splits=k, shuffle=True, random_state=None)
  models = []
  maxAccuracy = 0
  maxParams = {}
  model = XGBClassifier()
  
  
  for train_index, test_index in kfold.split(xTrain):
    params = {"learning_rate"    : random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ]),
    "max_depth"        : random.choice([ 3, 4, 5, 6, 8, 10, 12, 15]),
    "min_child_weight" : random.choice([ 1, 3, 5, 7 ]),
    "gamma"            : random.choice([ 0.0, 0.1, 0.2 , 0.3, 0.4 ]),
    "colsample_bytree" : random.choice([ 0.3, 0.4, 0.5 , 0.7 ]),
    "objective" : 'binary:logistic',
    "use_label_encoder" : False }

    x_Train, x_Test = xTrain[train_index], xTrain[test_index]
    y_Train, y_Test = yTrain[train_index], yTrain[test_index]
    model.set_params(**params)
    model.fit(x_Train, y_Train)
    yPredTest = model.predict(x_Test)
    accuracy = accuracy_score(y_Test, yPredTest)
    print("Accuracy score Test ", accuracy)
    print(params)
    if accuracy > maxAccuracy:
      maxParams = model.get_xgb_params()
      maxAccuracy = accuracy

  print(maxParams)
  model.set_params(**maxParams)
  model.fit(xTrain, yTrain)
  yPredTest = model.predict(xTest)

  accuracy = accuracy_score(yTest, yPredTest)
  print("Accuracy score Test ", accuracy_score(yPredTest, yTest))
  print("Confusion matrix Test ", confusion_matrix(yPredTest, yTest))
  cm = confusion_matrix(yPredTest, yTest)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Confusion matrix : Normalised Test ", cm)

def xgBoost():
  trainData, testData, dtypes = load_csv()
  xTrain, yTrain, xTest, yTest = pre_processing(trainData, testData, dtypes)
  model = XGBClassifier()
  model.fit(xTrain, yTrain)
  yPredTrain = model.predict(xTrain)
  yPredTest = model.predict(xTest)

  accuracy = accuracy_score(yTrain, yPredTrain)
  print("Accuracy score Train ", accuracy_score(yPredTrain, yTrain))
  print("Confusion matrix Train ", confusion_matrix(yPredTrain, yTrain))
  cm = confusion_matrix(yPredTrain, yTrain)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Confusion matrix : Normalised Train ", cm)
  print("\n")
  accuracy = accuracy_score(yTest, yPredTest)
  print("Accuracy score Test ", accuracy_score(yPredTest, yTest))
  print("Confusion matrix Test ", confusion_matrix(yPredTest, yTest))
  cm = confusion_matrix(yPredTest, yTest)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("Confusion matrix : Normalised Test ", cm)



def pre_processing(trainData, testData, dtypes):
  imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
  imputer.fit(trainData)
  trainData=imputer.transform(trainData)
  testData=imputer.transform(testData)
  for ind,i in enumerate(dtypes):
    if i=="object":
      encoder = OrdinalEncoder()
      encoder.fit(trainData[:,ind].reshape(-1, 1))
      trainData[:,ind] = encoder.transform(trainData[:,ind].reshape(-1, 1)).flatten()
      testData[:,ind] = encoder.transform(testData[:,ind].reshape(-1, 1)).flatten()

  xTrain = trainData[:,:-1]
  yTrain = trainData[:, [-1]].flatten()
  xTest = testData[:,:-1]
  yTest = testData[:, [-1]].flatten()
  return xTrain, yTrain.astype(int), xTest, yTest.astype(int)

def load_csv():
  trainData = pd.read_csv('./adult.data', skipinitialspace=True, header=None)
  dtypes = np.array([str(i) for i in trainData.dtypes])
  trainData = trainData.to_numpy(dtype=object)
  testData = pd.read_csv('./adult.test', skipinitialspace=True, header=None).to_numpy(dtype=object)
  return trainData, testData, dtypes

# xgBoost()
# xgBoostKfold(10)