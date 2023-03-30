from scipy.special import softmax
import numpy as np
import random

import config
from exp.exp import Exp
from lib_classifier.mlp import MLP
from lib_classifier.rf import RF


class ExpClassMemInfer(Exp):
    def __init__(self, args):
        super(ExpClassMemInfer, self).__init__(args)

        self.attack_train_data, self.attack_train_label = None, None
        self.attack_train_data_basic, self.attack_test_data_basic = None, None
        self.attack_test_data, self.attack_test_label = None, None
        self.attack_train_classes, self.attack_test_classes = None, None
        self.basic_similarity = None
        self.determine_attack_model()

    def determine_attack_model(self):
        if self.args['attack_model'] == 'MLP':
            self.attack_model = MLP()
        elif self.args['attack_model'] == 'RF':
            self.attack_model = RF()

    def train_attack_model(self):
        self.logger.info('training attack model')
        save_name = config.MODEL_PATH + "attack_model/" + "_".join((self.args['dataset_name'], self.args['target_model'], self.args['attack_model']))
        if self.args["is_normalize_similarity"]:
            self.attack_train_data = softmax(self.attack_train_data)

        if self.args['attack_model'] == 'MLP':
            self.attack_model = MLP()
            self.attack_model.train_model(self.attack_train_data, self.attack_train_label, save_name=save_name)
        elif self.args['attack_model'] == 'RF':
            self.attack_model = RF()
            self.attack_model.train_model(self.attack_train_data, self.attack_train_label, min_samples_leaf=10, save_name=save_name)

    def load_attack_model(self):
        self.attack_model.load_model(self.attack_model_save_path)

    def evaluate_attack_model(self):
        self.logger.info('testing attack model')
        
        if self.args["is_normalize_similarity"]:
            self.attack_test_data = softmax(self.attack_test_data)
        self.attack_acc = self.attack_model.test_model_acc(self.attack_train_data, self.attack_train_label)
        self.attack_auc = self.attack_model.test_model_auc(self.attack_train_data, self.attack_train_label)
        self.attack_f1_score, self.attack_recall, self.attack_precision, self.attack_FPR = self.attack_model.test_model_metrics(self.attack_train_data, self.attack_test_label)

        
        if self.args['target_model'] == "siamesenet":
            self.attack_class_acc, self.true_similarity, self.false_similarity = self.attack_model.test_model_class_acc(self.attack_train_data, self.attack_train_label, self.attack_train_classes, self.basic_similarity)
            self.logger.info("attack model train acc %s, auc %s, class acc %s, f1_score %s, recall %s, precision %s, FPR %s" % (self.attack_acc, self.attack_auc, self.attack_class_acc, self.attack_f1_score, self.attack_recall, self.attack_precision, self.attack_FPR))
        else:
            self.logger.info("attack model train acc %s, auc %s, f1_score %s, recall %s, precision %s, FPR %s" % (self.attack_acc, self.attack_auc, self.attack_f1_score, self.attack_recall, self.attack_precision, self.attack_FPR))

        self.attack_acc = self.attack_model.test_model_acc(self.attack_test_data, self.attack_test_label)
        self.attack_auc = self.attack_model.test_model_auc(self.attack_test_data, self.attack_test_label)
        self.attack_f1_score, self.attack_recall, self.attack_precision, self.attack_FPR = self.attack_model.test_model_metrics(self.attack_test_data, self.attack_test_label)
        
        if self.args['target_model'] == "siamesenet":
            self.attack_class_acc, self.true_similarity, self.false_similarity = self.attack_model.test_model_class_acc(self.attack_test_data, self.attack_test_label, self.attack_test_classes, self.basic_similarity)

            self.logger.info("attack model test acc %s, auc %s, class acc %s, f1_score %s, recall %s, precision %s, FPR %s" % (self.attack_acc, self.attack_auc, self.attack_class_acc, self.attack_f1_score, self.attack_recall, self.attack_precision, self.attack_FPR))
            self.logger.info("true_similarity %s, false_similarity%s, mean_similarity %s" % (self.true_similarity, self.false_similarity, np.mean(self.basic_similarity)))
        else:
            self.logger.info("attack model test acc %s, auc %s, f1_score %s, recall %s, precision %s, FPR %s" % (self.attack_acc, self.attack_auc, self.attack_f1_score, self.attack_recall, self.attack_precision, self.attack_FPR))

    def visualize(self):
        embeddings, labels = np.concatenate((self.attack_train_data, self.attack_test_data), axis=0), np.concatenate((self.attack_train_label, self.attack_test_label), axis=0)
        if self.args['is_use_image_similarity']:
            self.plt.tsne(embeddings, labels, title= str(round(self.attack_auc, 3)), out="_".join((self.args['dataset_name'], self.args['target_model'], 'tsne_anchor.pdf')))
        else:
            self.plt.tsne(embeddings, labels , title= str(round(self.attack_auc, 3)), out="_".join((self.args['dataset_name'], self.args['target_model'], 'tsne_basic.pdf')))

    def _calculate_Cu(self, embeddings):
        Cu = []
        for i in range(embeddings.shape[0]):
            c=np.mean(embeddings[i], axis=0)
            dist=np.sum(np.linalg.norm(np.stack((embeddings[i][j], c))) for j in range(embeddings.shape[1]))
            Cu.append(dist/embeddings.shape[1])
        return np.array(Cu).reshape(-1, 1)

    def _calculate_Pu(self, embeddings):
        Pu, pair_wise_dist = [], []
        for i in range(embeddings.shape[0]):
            for j in range(embeddings.shape[1]-1):
                for k in range(j+1, embeddings.shape[1]):
                    pair_wise_dist.append(np.linalg.norm(np.stack((embeddings[i][j], embeddings[i][k]))))
                dist=np.sum(pair_wise_dist)/len(pair_wise_dist)
            Pu.append(dist)
        return np.array(Pu).reshape(-1, 1)

    def _PuCu_calculation(self, train_embeddings, test_embeddings):
        if 'Cu' in self.args['embedding_metric']:
            train_score_Cu = self._calculate_Cu(train_embeddings)
            test_score_Cu = self._calculate_Cu(test_embeddings)
        if 'Pu' in self.args['embedding_metric']:
            train_score_Pu = self._calculate_Pu(train_embeddings)
            test_score_Pu = self._calculate_Pu(test_embeddings)
        if self.args['embedding_metric'] == "Cu":
            train_score = train_score_Cu
            test_score = test_score_Cu
        elif self.args['embedding_metric'] == "Pu":
            train_score = train_score_Pu
            test_score = test_score_Pu
        else:
            train_score = np.concatenate((train_score_Cu, train_score_Pu), axis=1)
            test_score = np.concatenate((test_score_Cu, test_score_Pu), axis=1)
        return train_score, test_score

    def _perturbe_scores(self, scores):
        ret_scores = np.zeros_like(scores)
        for i, score in enumerate(scores):
            ret_scores[i] = score + np.random.laplace(loc=0.0, scale=self.args['noise_std'], size=score.size)
        return ret_scores

    def _memguard_random_perturbe_scores(self, scores):
        """
        Arbitrary changes to the confidence vector so long as it does not change the label.
        Args:
        scores: confidence vectors as 2d numpy array
        Returns: 2d scores protected by memguard.
        """
        n_classes = scores.shape[1]
        epsilon = 1e-3
        on_score = (1. / n_classes) + epsilon
        off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
        predicted_labels = np.argmax(scores, axis=-1)
        defended_scores = np.ones_like(scores) * off_score
        defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
        return defended_scores

    def _memguard_iterative_perturbe_scores(self, scores, epsilon=0.3):
        """
        Iteratively finding the noise perturbation for the maximum posterior value.
        Args:
        scores: confidence vectors as 2d numpy array
        Returns: 2d scores protected by memguard.
        """
        n_classes = scores.shape[1]
        predicted_labels = np.argmax(scores, axis=-1)
        defended_scores = np.ones_like(scores)

        for i in range(len(defended_scores)):
            r = np.zeros(scores.shape[1])
            r[0] = random.uniform(0, 1)
            for j in range(1, 5):
                r[j] = random.uniform(0, 1-np.sum(r))
            defended_scores[i, predicted_labels] = softmax(scores,axis=-1)[i, predicted_labels] - max(r)

        return defended_scores