import pandas as pd
import kNN
import unittest
import UnitTesting

colnames =["col1", "col2", "col3", "col4", "label"]
dataLearning = pd.read_csv("iris.data.learning", names=colnames)
dataTest = pd.read_csv("iris.data.test", names=colnames)

dataList = dataLearning.values.tolist()
testList = dataTest.values.tolist()

test = kNN.kNN(121, dataList)

print("\n       PREDICT  FUNCTION RESULT:        \n")
print(test.predict(testList))
print("\n       SCORE  FUNCTION RESULT:        \n")
print(test.score(testList))
