import logging
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


class RF:
    def __init__(self):
        pass

    def train_model(self, train_x, train_y, min_samples_leaf=10, save_name=None):
        self.model = RandomForestClassifier(max_leaf_nodes=10, random_state=0)
        self.model.fit(train_x, train_y)

        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)

        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        return accuracy_score(test_y, pred_y)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])

    def test_model_class_acc(self, test_x, test_y, test_class, basic_distance):
        pred_y = self.model.predict(test_x)
        classes = np.unique(test_class)
        test_class_y = []
        pred_class_y = []

        true_indices = np.where(test_y == pred_y)
        true_distance = basic_distance[true_indices]
        false_indices = np.where(test_y != pred_y)
        false_distance = basic_distance[false_indices]

        print("true distance", np.mean(true_distance), "false_distance", np.mean(false_distance))

        for c in classes:
            indices = np.where(test_class == c)
            test_class_y.append(np.argmax(np.bincount(test_y[indices].astype(np.int64))))
            pred_class_y.append(np.argmax(np.bincount(pred_y[indices].astype(np.int64))))
        return accuracy_score(test_class_y, pred_class_y), np.mean(true_distance), np.mean(false_distance)

    def test_distance_relation(self, test_x, test_y, test_class, basic_distance):
        pred_y = self.model.predict(test_x)

        classes = np.unique(test_class)
        test_class_y = []
        pred_class_y = []
        true_indices = np.where(test_y == pred_y)
        true_distance = basic_distance[true_indices]
        false_indices = np.where(test_y != pred_y)
        false_distance = basic_distance[false_indices]

        for c in classes:
            indices = np.where(test_class == c)
            test_class_y.append(np.argmax(np.bincount(test_y[indices].astype(np.int64))))
            pred_class_y.append(np.argmax(np.bincount(pred_y[indices].astype(np.int64))))
        return accuracy_score(test_class_y, pred_class_y)
