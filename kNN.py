import scipy.spatial.distance as sp


class kNN:
    def __init__(self, k, *data):
        self.k = k
        self.data = data




    def predict(self, *test):

        j = 0
        dist = []
        numberOfOcurrencies = {}
        result = []

        while(j < len(test[0])):
            dist.clear()
            numberOfOcurrencies.clear()
            i = 0
            m = 0
            max = 0
            prediction = ''

            while(i < len(self.data[0])):
                x = [[self.data[0][i][0], self.data[0][i][1], self.data[0][i][2], self.data[0][i][3]], [test[0][j][0], test[0][j][1], test[0][j][2], test[0][j][3]]]
                y = sp.pdist(x, 'euclidean')
                dist.append([y, self.data[0][i][4]])
                i += 1

            dist.sort(key=lambda dist:dist[0])

            while(m < len(dist) and m < self.k):
                if dist[m][1] in numberOfOcurrencies:
                    temporary = numberOfOcurrencies[dist[m][1]] + 1
                    del numberOfOcurrencies[dist[m][1]]
                    numberOfOcurrencies[dist[m][1]] = temporary
                else:
                    numberOfOcurrencies[dist[m][1]] = 1

                m += 1

            for item in numberOfOcurrencies:
                if numberOfOcurrencies[item] > max:
                    max = numberOfOcurrencies[item]
                    prediction = item

            result.append(prediction)
            j += 1

        return result

    def score(self, *test):
        j = 0
        dist = []
        numberOfOcurrencies = {}
        result = []

        while(j < len(test[0])):
            dist.clear()
            numberOfOcurrencies.clear()
            i = 0
            m=0
            max = 0
            prediction = ''

            while(i < len(self.data[0])):
                x = [[self.data[0][i][0], self.data[0][i][1], self.data[0][i][2], self.data[0][i][3]], [test[0][j][0], test[0][j][1], test[0][j][2], test[0][j][3]]]
                y = sp.pdist(x, 'euclidean')
                dist.append([y, self.data[0][i][4]])
                i += 1

            dist.sort(key=lambda dist:dist[0])

            while(m < len(dist) and m < self.k):
                if dist[m][1] in numberOfOcurrencies:
                    temporary = numberOfOcurrencies[dist[m][1]] + 1
                    del numberOfOcurrencies[dist[m][1]]
                    numberOfOcurrencies[dist[m][1]] = temporary
                else:
                    numberOfOcurrencies[dist[m][1]] = 1

                m += 1

            for item in numberOfOcurrencies:
                if numberOfOcurrencies[item] > max:
                    max = numberOfOcurrencies[item]
                    prediction = item

            result.append(prediction)
            j += 1

        i = 0
        goodAnswers = 0
        totalAnswers = 0

        while i < len(test[0]):
            if test[0][i][4] == result[i]:
                totalAnswers += 1
                goodAnswers += 1
            else:
                totalAnswers += 1
            i += 1
        return goodAnswers/totalAnswers

