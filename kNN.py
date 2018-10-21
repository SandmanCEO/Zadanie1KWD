import scipy.spatial.distance as sp

class kNN:
    def __init__(self, k, *data):
        self.k = k
        self.data = data


    def predict(self, *test):

        j = 0
        result = []

        while(j < len(test[0])):
            del result[:]
            i = 0
            while(i < len(self.data[0])):
                x = [[self.data[0][i][0], self.data[0][i][1], self.data[0][i][2], self.data[0][i][3]], [test[0][j][0], test[0][j][1], test[0][j][2], test[0][j][3]]]
                y = sp.pdist(x, 'euclidean')
                result.append([y, self.data[0][i][4]])
                i += 1

            result.sort(key=lambda result:result[0])
            j += 1
            print(result)

