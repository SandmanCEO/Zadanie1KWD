import pandas as pd
import kNN
import unittest

colnames =["col1", "col2", "col3", "col4", "label"]
dataLearning = pd.read_csv("iris.data.learning", names=colnames)
dataTest = pd.read_csv("iris.data.test", names=colnames)

dataList = dataLearning.values.tolist()
testList = dataTest.values.tolist()

test = kNN.kNN(123, dataList)

class TestkNN(unittest.TestCase):

    def test_predict(self):
        self.assertIsNotNone(test.predict(testList))

    def test_score(self):
        self.assertGreaterEqual(test.score(testList), 0)
        self.assertLessEqual(test.score(testList), 1)

"""if __name__ == '__main__':
    unittest.main()"""
