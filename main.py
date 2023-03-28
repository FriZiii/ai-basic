from math import sqrt
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class Data:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.data = [list(x) for x in self.df.values]
        self.normalized = []
        self.training = []
        self.validation = []
    
    def randomize(self):
        """
        The function shuffles the dataset.
        """
        shuffled = []
        for x in range(len(self.data)):
            i = random.randint(0, len(self.data)-1)
            shuffled.append(self.data[i])
            self.data.pop(i)
        self.data = shuffled

    def normalize(self):
        """
        The function normalizes the data from the array of arrays to the range 0.1
        """
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

class KNN:
    def __init__(self, training_set):
        self.training = training_set
    
    def _metric(self, x, y, m):
        """
        The function returns the Minkowski distances between two vectors.
        The arguments x and y are a vector, and m is an integer denoting the degree of the metric.
        """
        if len(x) != len(y):
            raise ValueError("The given vectors are not of the same length")
        addition = 0
        for i in range(len(x)):
            if isinstance(x[i], (int, float)):
                addition += pow(abs(x[i]-y[i]), m)

        return pow(addition, 1/m)
    
    def classify(self, value, k, m):
        """
        The function, based on the knn algorithm, determines to which of the classes the given object belongs.
        The arguments value is the object to be classified, k defines the number of nearest neighbors, and m the degree of the metric.
        In the event of an undefined situation, logs are saved.
        """
        lengths = []
        for x in self.training:
            lengths.append([self._metric(value, x, m), x[-1:]])
        
        k_nearest = sorted(lengths, key=lambda x:x[0])[:k]
        frequency = {}

        for element in k_nearest:
            name = element[1][0]
            if name in frequency:
                frequency[name] += 1
            else:
                frequency[name] = 1

        max_value = max(frequency.values())
        max_keys = [k for k, v in frequency.items() if v == max_value]
        if len(max_keys) == 1:
            return max_keys[0]
        else:
            with open("logs.txt", "a") as logs:
                logs.write(f"\tA random value was assumed. For {value}, there was a draw for {frequency}\n")
            return random.choice(max_keys)
    
    def accuracy(self, validation_set, k, m):
        """
        The function calculates the accuracy of the algorithm for the validation set and returns it as a percentage.
        The function takes arrays of arrays as a validation set, and k defining the number of nearest neighbors and m as the degree of the metric.
        """
        counter = 0
        denominator = len(validation_set)
        for x in validation_set:
            if self.classify(x, k, m) == x[-1:][0]:
                counter+=1
        return((counter/denominator)*100)


def find_best_parameters(maxK, maxM, n):
    """  
    The function returns a sorted array of arrays from least to greatest accuracy, for different parameters k and m
    
    maxK denotes the right end of the interval for the parameter k of the nearest neighbors.
    maxM denotes the right end of the interval parameter for the m metric parameter.
    n is the amount n iterations for each parameter
    """
    if os.path.exists("logs.txt"):
        os.remove("logs.txt")
    result = []
    for m in tqdm(range(1, maxM)):
        for k in range(1, maxK):
            sum = 0
            xn = []
            for itr in range(n):
                iris = Data('E:\Studia\AI\\ai-basic\iris.csv')
                iris.randomize()
                iris.normalize()
                iris.distribution(30)

                with open("logs.txt", "a") as logs:
                    logs.write(f"Call {itr}: with k = {k} and m = {m}\n")

                knn = KNN(iris.training)
                accuracy = knn.accuracy(iris.validation, k, m)
                xn.append(accuracy)
                sum += accuracy

            average = sum/n
            standard_deviation = 0
            for i in xn:
                standard_deviation += pow(i - average,2)
            result.append([m,k,round(average,2), round(sqrt(standard_deviation/n),2)])
        result = sorted(result, key=lambda x: x[2]-x[3])
    return result

def graph(data, k):
    """
    The function generates a graph for k vectors of accuracy
    """
    top_k = data[-k:]
    print(top_k)
    labels = [f"{x[0]}, {x[1]}" for x in top_k]
    values = [x[2] for x in top_k]
    std_devs = [x[3] for x in top_k]

    values, labels, std_devs = zip(*sorted(zip(values, labels, std_devs), reverse=True))

    plt.bar(labels, values, yerr=std_devs)
    plt.xticks(rotation=90)
    plt.show()

data = find_best_parameters(10,10,25)
greatest_accuracy = data[-1:]
graph(data, 15)