import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


class MLP:
    def __init__(self):
        pass

    def scaler_data(self, data):
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data

    def train_model(self, train_x, train_y, save_name=None):
        self.model = MLPClassifier()
        self.model.fit(train_x, train_y)

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
    
    def test_model_metrics(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        recall = recall_score(test_y, pred_y, average='binary')
        precision = precision_score(test_y, pred_y, average='binary')
        fscore = f1_score(test_y, pred_y, average='binary')
        CM = confusion_matrix(test_y, pred_y)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        FPR = FP/(FP+TN)
        return fscore, recall, precision, FPR


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
