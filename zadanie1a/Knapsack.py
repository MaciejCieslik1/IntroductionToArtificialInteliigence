import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
import time
import gc


# Author: Maciej Cieślik 


class Knapsack:

    def __init__(self, profits, weights, capacity):
        self.profits = profits
        self.weights = weights
        self.capacity = capacity

    def solveKnapsackBruteForce(self):
        allSubsets = list(itertools.product([0, 1], repeat=len(self.weights)))
        bestSubset = []
        bestProfit = 0
        bestWeight = 0
        for currentSubset in allSubsets:
            currentProfit = 0
            currentWeight = 0
            for index, value in enumerate(currentSubset):
                if value == 1 and currentWeight + self.weights[index] <= self.capacity:
                    currentProfit += self.profits[index]
                    currentWeight += self.weights[index]
            if currentProfit > bestProfit:
                bestProfit = currentProfit
                bestWeight = currentWeight
                bestSubset = currentSubset
        choosenIndexes = [index for index, value in enumerate(bestSubset) if value]
        return choosenIndexes, bestProfit, bestWeight

    def solveKnapsackPwRatio(self):
        pwRatios = self.profits / self.weights
        tuples = [(index, self.profits[index], self.weights[index], pwRatios[index]) for index in range(len(self.profits))]
        sortedTuples = sorted(tuples, key=lambda x: (-x[3], -x[1]))
        currentWeight = 0
        currentProfit = 0
        choosenIndexes = []
        for currentTuple in sortedTuples:
            if currentWeight + currentTuple[2] <= self.capacity:
                choosenIndexes.append(currentTuple[0])
                currentProfit += currentTuple[1]
                currentWeight += currentTuple[2]
            else:
                break
        return choosenIndexes, currentProfit, currentWeight


def formatResult(profits, weights, capacity, methodID):
    # ID 0: run solveKnapsackBruteForce
    # ID 1: run solveKnapsackPwRatio

    knapsack = Knapsack(profits, weights, capacity)
    if methodID == 0:
        indexes, profit, weight = knapsack.solveKnapsackBruteForce()
        print("Brute Force method")
    else:
        indexes, profit, weight = knapsack.solveKnapsackPwRatio()
        print("Profit/weight ratio method")
    print("Choosen Indexes:", end=' ')
    print(indexes)
    print("Profit:", end=' ')
    print(profit)
    print("Weight:", end=' ')
    print(weight)


def generatePlot():
    profits = []
    weights = []
    CAPACITY = 100
    numberOfBeginningItems = 10
    numberOfAddedItems = 15
    for _ in range(numberOfBeginningItems):
        profits.append(random.randint(1, 20))
        weights.append(random.randint(1, 20))
    numberOfElements = len(profits)
    times = []
    for _ in range(numberOfAddedItems):
        knapsack = Knapsack(profits, weights, CAPACITY)
        gc.disable()
        startTime = time.time()
        knapsack.solveKnapsackBruteForce()
        endTime = time.time()
        gc.enable()
        times.append(endTime - startTime)
        profits.append(random.randint(1, 20))
        weights.append(random.randint(1, 20))
        numberOfElements += 1
    values = [i for i in range (numberOfBeginningItems, numberOfElements, 1)]
    plt.plot(values, times, marker='o', linestyle='-', color='b', label="t(n)")
    plt.title("Knapsack problem - Brute Force method")
    plt.xlabel("Number of items")
    plt.ylabel("Time [s]")
    plt.legend()
    plt.savefig('plot.png')
    plt.show()

if __name__ == "__main__":
    #generatePlot()
    weights = np.array([8, 3, 5, 2])
    capacity = 9
    profits = np.array([16, 8, 9, 6])
    formatResult(profits, weights, capacity, 0)
    formatResult(profits, weights, capacity, 1)