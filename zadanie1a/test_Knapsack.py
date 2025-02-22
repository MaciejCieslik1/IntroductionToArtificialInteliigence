from Knapsack import Knapsack
import numpy as np


# Author: Maciej Cieślik 


# tests for bruce force method

def testKnapsackBruteForce1():
    profits = [10]
    weights = [5]
    capacity = 5
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([0], 10, 5)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce2():
    profits = [10, 20, 30]
    weights = [5, 10, 15]
    capacity = 25
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([1, 2], 50, 25)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce3():
    profits = [20, 30, 40]
    weights = [10, 20, 30]
    capacity = 15
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([0], 20, 10)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce4():
    profits = [10, 20, 30]
    weights = [15, 25, 35]
    capacity = 10
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([], 0, 0)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce5():
    profits = []
    weights = []
    capacity = 10
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([], 0, 0)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce6():
    profits = [50]
    weights = [60]
    capacity = 50
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([], 0, 0)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce7():
    profits = [10, 10, 10]
    weights = [5, 5, 5]
    capacity = 10
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([1, 2], 20, 10)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce8():
    profits = [15, 25, 10, 40, 20, 30]
    weights = [2, 3, 4, 5, 1, 4]
    capacity = 7
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([0, 4, 5], 65, 7)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce9():
    profits = [5, 10, 15, 7, 8, 12]
    weights = [1, 2, 3, 2, 1, 2]
    capacity = 5
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([1, 4, 5], 30, 5)
    assert knapsack.solveKnapsackBruteForce() == expected


def testKnapsackBruteForce10():
    profits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    capacity = 15
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([6, 7], 15, 15)
    assert knapsack.solveKnapsackBruteForce() == expected

# tests for pw ratio method

def testKnapsackPwRatio1():
    profits = np.array([10])
    weights = np.array([5])
    capacity = 5
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([0], 10, 5)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio2():
    profits = np.array([10, 20, 30])
    weights = np.array([5, 10, 15])
    capacity = 25
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([2, 1], 50, 25)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio3():
    profits = np.array([20, 30, 40])
    weights = np.array([10, 20, 30])
    capacity = 15
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([0], 20, 10)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio4():
    profits = np.array([10, 20, 30])
    weights = np.array([15, 25, 35])
    capacity = 10
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([], 0, 0)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio5():
    profits = np.array([])
    weights = np.array([])
    capacity = 10
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([], 0, 0)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio6():
    profits = np.array([50])
    weights = np.array([60])
    capacity = 50
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([], 0, 0)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio7():
    profits = np.array([10, 10, 10])
    weights = np.array([5, 5, 5])
    capacity = 10
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([0, 1], 20, 10)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio8():
    profits = np.array([15, 25, 10, 40, 20, 30])
    weights = np.array([2, 3, 4, 5, 1, 4])
    capacity = 7
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([4, 1, 0], 60, 6)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio9():
    profits = np.array([5, 10, 15, 7, 8, 12])
    weights = np.array([1, 2, 3, 2, 1, 2])
    capacity = 5
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([4, 5, 1], 30, 5)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio10():
    profits = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    capacity = 15
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([9, 4], 15, 15)
    assert knapsack.solveKnapsackPwRatio() == expected


def testKnapsackPwRatio11():
    profits = np.array([15, 20, 25])
    weights = np.array([3, 4, 5])
    capacity = 7
    knapsack = Knapsack(profits, weights, capacity)
    expected = ([0, 1], 35, 7)
    assert knapsack.solveKnapsackPwRatio() == expected