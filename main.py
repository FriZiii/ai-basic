import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random

class Data:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.data = [list(x) for x in self.df.values]
        self.normalized = []
        self.training = []
        self.validation = []
    
    def randomize(self):
        shuffled = []
        for x in range(len(self.data)):
            i = random.randint(0, len(self.data)-1)
            shuffled.append(self.data[i])
            self.data.pop(i)
        self.data = shuffled

    def normalize(self):
        minmax = []
        for i in range(len(self.data[0])):
            min_val = None
            max_val = None
            for j in range(len(self.data)):
                if isinstance(self.data[j][i], (int, float)):
                    if min_val is None or self.data[j][i] < min_val:
                        min_val = self.data[j][i]
                    if max_val is None or self.data[j][i] > max_val:
                        max_val = self.data[j][i]
            minmax.append([min_val, max_val])
        
        for i in range(len(self.data[0])):
            for j in range(len(self.data)):
                if isinstance(self.data[j][i], (int, float)):
                    self.data[j][i] = (self.data[j][i] - minmax[i][0])/(minmax[i][1]-minmax[i][0])

    def distribution(self, procent):
        """
        The method splits a set into 2 smaller sets. 
        The argument is a percentage of the size of the validation set.
        """
        if procent < 0 or procent > 100:
            raise ValueError("The percentage must be between 0 and 100.")
    
        n = len(self.data)
        k = int(n * procent / 100)
        self.validation = self.data[:k]
        self.training = self.data[k:]

    def show(self, var):
        a = ""
        for d, i in enumerate(var):
            a += f"{d}: {i} \n"
        print(a)



iris = Data('E:\Studia\AI\\ai-basic\iris.csv')
iris.randomize()
iris.normalize()
iris.distribution(70)

iris.show(iris.validation)
iris.show(iris.training)

#df = pd.DataFrame(iris.data, columns=["sepal.length","sepal.width","petal.length","petal.width","variety"])
#df_validation = pd.DataFrame(iris.validation, columns=["sepal.length","sepal.width","petal.length","petal.width","variety"])
#df_training = pd.DataFrame(iris.training, columns=["sepal.length","sepal.width","petal.length","petal.width","variety"])
#figure, axis = plt.subplots(1, 3)
#sb.pairplot(df , hue="variety")
#sb.pairplot(df_training , hue="variety")
#sb.pairplot(df_validation , hue="variety")
#plt.show()