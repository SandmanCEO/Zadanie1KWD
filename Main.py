import pandas as pd
import numpy as np
import kNN
import scipy.spatial.distance as sp

colnames =["col1", "col2", "col3", "col4", "label"]
dataLearning = pd.read_csv("iris.data.learning", names=colnames)
dataTest = pd.read_csv("iris.data.test", names=colnames)

dataList = dataLearning.values.tolist()
testList = dataTest.values.tolist()

test = kNN.kNN(1, dataList)

test.predict(testList)