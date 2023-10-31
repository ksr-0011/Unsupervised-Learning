import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

f = open("data.npy", "rb")
data = np.load(f, allow_pickle=True)

def euclidean_distance(x1, x2):
    # to match shape
    x1 = x1.flatten()
    x2 = x2.flatten()
    dist = np.sqrt(np.sum((x2 - x1)**2))
    return dist

def manhattan_distance(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    dist = np.sum(np.abs(x2 - x1))
    return dist

def cosine_distance(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    dist = np.dot(x1, x2)
    dist /= (np.sqrt(np.sum(x1**2)) * np.sqrt(np.sum(x2**2)))
    dist = 1 - dist
    return dist

class KNNModel():
    def __init__(self, k = 12, distance_metric = 'manhattan', encoder_type = 'vit'):
        self.k = k
        self.distance_metric = distance_metric
        self.encoder_type = encoder_type

    def fit(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test 
        self.y_test = y_test    

    def pred(self):
        y_pred = []
        
        for x in self.X_test:
            if(self.distance_metric == 'euclidean'):
                distance_arr = [euclidean_distance(X_train, x) for X_train in self.X_train]
            elif(self.distance_metric == 'manhattan'):
                distance_arr = [manhattan_distance(X_train, x) for X_train in self.X_train]
            elif(self.distance_metric == 'cosine'):
                distance_arr = [cosine_distance(X_train, x) for X_train in self.X_train]       
            sorted_dist = np.argsort(distance_arr)
            k_nearest_indices = sorted_dist[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            count = {}
            for label in k_nearest_labels:
                count[label] = count.get(label, 0) + 1
            mode = max(count, key=count.get)
            y_pred.append(mode)
        self.y_pred = y_pred
        return y_pred   

    def acc(self):
        accuracy = accuracy_score(self.y_test,self.y_pred)
        # print(f"The accuracy is {accuracy}")
        return accuracy
        

    def precision(self):
        precision = precision_score(self.y_test,self.y_pred, average='micro')
        # print(f"The precision is {precision}")
        return precision

    def recall(self):
        recall = recall_score(self.y_test,self.y_pred, average='micro')
        # print(f"The recall is {recall}")
        return recall

    def f1(self):
        f1 = f1_score(self.y_test,self.y_pred, average='micro')
        # print(f"The f1 score is {f1}") 
        return f1


def main_test(test_file, k = 12, encoder_type = 'vit', distance = 'manhattan'):
    test_data = np.load(test_file, allow_pickle=True)
    classifier = KNNModel()
    # print(encoder_type)
    if(encoder_type == 'vit'):
        X_train = data[: , 2]
        y_train = data[: ,3]
        X_test = test_data[: , 2]
        y_test = test_data[: , 3]
    
    elif(encoder_type == 'resnet'):
        X_train = data[: , 1]
        y_train = data[: , 3]
        X_test = test_data[: , 1]
        y_test = test_data[: , 3]

    classifier.fit(X_train, y_train, X_test, y_test)
    classifier.pred()   
    accuracy = classifier.acc()
    precision = classifier.precision()
    recall = classifier.recall()
    f1 = classifier.f1()

     # print in a nice table all the metrics
    print("| Metric     | Score       |")
    print("|------------|-------------|")
    print(f"| Accuracy   | {accuracy}   |")
    print(f"| F1 Score   | {f1}   |")
    print(f"| Precision  | {precision}   |")
    print(f"| Recall     | {recall}   |") 