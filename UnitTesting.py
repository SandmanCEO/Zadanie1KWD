import pandas as pd
import kNN
import unittest
import Main

class TestkNN(unittest.TestCase):

    def test_predict(self):
        self.assertIsNotNone(Main.test.predict(Main.testList))

    def test_score(self):
        self.assertGreaterEqual(Main.test.score(Main.testList), 0)
        self.assertLessEqual(Main.test.score(Main.testList), 1)

if __name__ == '__main__':
    print("\n       UNIT TESTS EXECUTION RESULT:        \n")
    unittest.main()
