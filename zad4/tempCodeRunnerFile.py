from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import math


def entropy_func(class_count: int, num_samples: int) -> float:
    # count entropy value for one class
    ratio = class_count / num_samples
    return - ratio * math.log2(ratio)


class Group:
    def __init__(self, classes: list[int]):
        self.group_classes = classes
        self.entropy = self.group_entropy()

    def __len__(self) -> int:
        return len(self.group_classes)

    def group_entropy(self) -> float:
        # count sum of entropy values for all classes in group
        all_samples = sum(self.group_classes)
        entropy_sum = 0
        for single_class_count in self.group_classes:
            entropy_sum += entropy_func(single_class_count, all_samples)
        return entropy_sum


class Node:
    def __init__(self, split_feature, split_val, depth=None, child_node_a=None, child_node_b=None, val=None):
        self.split_feature = split_feature
        self.split_val = split_val
        self.depth = depth
        self.child_node_a = child_node_a
        self.child_node_b = child_node_b
        self.val = val

    def predict(self, data: list[float]) -> float:
        if self.child_node_a is None and self.child_node_b is None:
            return self.val
        if data[self.split_feature] <= self.split_val:
            return self.child_node_a.predict(data)
        else:
            return self.child_node_b.predict(data)


class DecisionTreeClassifier(object):
    def __init__(self, max_depth: int):
        self.depth = 0
        self.max_depth = max_depth
        self.tree = None
        self.min_classes_length = 2

    @staticmethod
    def get_split_entropy(group_a: Group, group_b: Group) -> float:
        # count entropy for splitted groups
        return group_a.group_entropy() + group_b.group_entropy()

    def get_information_gain(self, parent_group: Group, child_group_a: Group, child_group_b: Group) -> float:
        return parent_group.group_entropy() - self.get_split_entropy(child_group_a, child_group_b)

    def reorganize_to_group(self, classes: np.ndarray[int]) -> Group:
        classes_occurance = Counter(classes)
        group_classes = list(classes_occurance.values())
        return Group(group_classes)

    def split(self, data: np.ndarray, classes: np.ndarray, feature_pattern: float, feature_index: int) -> dict[str, any]:
        lower_rows = []
        higher_rows = []
        lower_classes = []
        higher_classes = []
        for index, row in enumerate(data):
            if row[feature_index] < feature_pattern:
                lower_rows.append(row)
                lower_classes.append(classes[index])
            else:
                higher_rows.append(row)
                higher_classes.append(classes[index])
        splitted_data = {"lower_rows": np.array(lower_rows),
                         "higher_rows": np.array(higher_rows),
                         "lower_classes": np.array(lower_classes),
                         "higher_classes": np.array(higher_classes)}
        return splitted_data

    def get_best_split(self, data: np.ndarray, classes: np.ndarray[int]) -> dict[str, any]:
        best_split_info = {}
        max_information_gain = float("-inf")
        for column_index in range(data.shape[1]):
            feature_column = data[:, column_index]
            for current_feature in feature_column:
                splitted_data = self.split(data, classes, current_feature, column_index)
                lower_classes = splitted_data["lower_classes"]
                higher_classes = splitted_data["higher_classes"]
                if len(splitted_data["lower_classes"]) > 0 and len(splitted_data["higher_classes"]) > 0:
                    parent_group = self.reorganize_to_group(classes)
                    lower_group = self.reorganize_to_group(splitted_data["lower_classes"])
                    higher_group = self.reorganize_to_group(splitted_data["higher_classes"])
                    information_gain = self.get_information_gain(parent_group, lower_group, higher_group)
                    if information_gain > max_information_gain:
                        max_information_gain = information_gain
                        best_split_info = splitted_data
                        best_split_info["information_gain"] = max_information_gain
                        best_split_info["feature_index"] = column_index
                        best_split_info["feature_value"] = current_feature
        return best_split_info

    def build_tree(self, data: np.ndarray, classes: np.ndarray[int], depth: int = 0) -> Node:
        if depth < self.max_depth:
            best_split = self.get_best_split(data, classes)
            left_subtree = self.build_tree(best_split["lower_rows"], best_split["lower_classes"], depth+1)
            right_subtree = self.build_tree(best_split["higher_rows"], best_split["higher_classes"], depth+1)
            return Node(best_split["feature_index"], best_split["feature_value"], depth, left_subtree, right_subtree)
        else:
            counter = Counter(classes)
            class_value = counter.most_common(1)[0][0]
            return Node(split_feature=None, split_val=None, depth=depth, val=class_value)

    def predict(self, data: np.ndarray):
        return self.tree.predict(data)


def test(test_size: float, random_state: int):
    iris = load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    NUMBER_OF_REPEATS = 10
    accuracies = []
    for depth in DEPTHS:
        sum_accuracy = 0
        for _ in range(NUMBER_OF_REPEATS):
            correct_counter = 0
            dc = DecisionTreeClassifier(depth)
            dc.tree = dc.build_tree(x_train, y_train)
            for sample, gt in zip(x_test, y_test):
                prediction = dc.predict(sample)
                if prediction == gt:
                    correct_counter += 1
            accuracy = correct_counter / len(y_test)
            sum_accuracy += accuracy
        average_accuracy = sum_accuracy / NUMBER_OF_REPEATS
        print(average_accuracy)
        accuracies.append(average_accuracy)
    generate_plot(DEPTHS, accuracies, test_size, random_state)


def generate_plot(dephts: list[int], accuracies: list[float], test_size: float, random_state: int):
    name = f"Impact of depth on prediction accuracy, test_ratio: {test_size} seed: {random_state}"
    plt.plot(dephts, accuracies, marker="o")
    plt.title(name)
    plt.xlabel("Depth of search")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(f"test_size_{test_size}_seed_{random_state}.png")
    plt.show()


# test(0.1, 123)

iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
correct_counter = 0
dc = DecisionTreeClassifier(4)
dc.tree = dc.build_tree(x_train, y_train)
for sample, gt in zip(x_test, y_test):
    prediction = dc.predict(sample)
    if prediction == gt:
        correct_counter += 1
accuracy = correct_counter / len(y_test)
print(accuracy)