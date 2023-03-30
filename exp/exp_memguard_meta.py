import logging

import numpy as np
from scipy.special import softmax

from exp.exp_class_mem_infer import ExpClassMemInfer
from lib_model.proto_net.proto_net import ProtoNet
from lib_dataset.task_dataset import TaskDataset, ProbeDataset, ProbeDatasetSort
from lib_dataset.transform import NWays, KShots, LoadData, RemapLabels
from lib_model.relation_net.relation_net import RelationNet


class ExpClassMemInferMeta(ExpClassMemInfer):
    def __init__(self, args):
        super(ExpClassMemInferMeta, self).__init__(args)
        self.logger = logging.getLogger('exp_class_mem_infer_meta')
        self.args = args
        self.determine_model()
        attack_acc, attack_auc, attack_precision, attack_recall, attack_f1_score, attack_FPR = [],[],[],[],[],[]

        for run in range(self.args['num_runs']):
            self.train_target_model()
            self.train_shadow_model()
            self.logger.info("**********Run %f**********" % run)
            self.probe_target_model()
            self.probe_shadow_model()
            if self.args['is_train_attack']:
                self.train_attack_model()
            else:
                self.load_attack_model()

            self.evaluate_attack_model()

            attack_acc.append(self.attack_acc)
            attack_auc.append(self.attack_auc)
            attack_precision.append(self.attack_precision)
            attack_recall.append(self.attack_recall)
            attack_f1_score.append(self.attack_f1_score)
            attack_FPR.append(self.attack_FPR)
            # self.visualize()

        self.attack_acc_avg = np.average(attack_acc)
        self.attack_auc_avg = np.average(attack_auc)
        self.attack_precision_avg = np.average(attack_precision)
        self.attack_recall_avg = np.average(attack_recall)
        self.attack_f1_score_avg = np.average(attack_f1_score)
        self.attack_acc_std = np.std(attack_acc)
        self.attack_auc_std = np.std(attack_auc)
        self.attack_precision_std = np.std(attack_precision)
        self.attack_recall_std = np.std(attack_recall)
        self.attack_f1_score_std = np.std(attack_f1_score)
        self.attack_FPR_avg = np.average(attack_FPR)
        self.attack_FPR_std = np.std(attack_FPR)

        if self.args['is_save_results']:
            self.save_results()

    def determine_model(self):
        if self.args['target_model']=="protonet":
            self.target_model = ProtoNet(self.args)
            self.shadow_model = ProtoNet(self.args)
        elif self.args['target_model'] == "relationnet":
            self.target_model = RelationNet(self.args)
            self.shadow_model = RelationNet(self.args)

        self.train_transforms = [
            NWays(self.target_train_mem_dset, self.args['way']),
            KShots(self.target_train_mem_dset, self.args['train_num_query'] + self.args['shot']),
            LoadData(self.target_train_mem_dset),
            RemapLabels(self.target_train_mem_dset),
            ]
        self.test_transforms = [
            NWays(self.target_test_dset, self.args['way']),
            KShots(self.target_test_dset, self.args['test_num_query'] + self.args['shot']),
            LoadData(self.target_test_dset),
            RemapLabels(self.target_test_dset),
        ]

    def train_target_model(self):
        self.logger.info('training target model')
        train_dset = TaskDataset(self.target_train_mem_dset, task_transforms=self.train_transforms)
        test_dset = TaskDataset(self.target_test_dset, task_transforms=self.test_transforms, num_tasks=self.args['test_num_task'])

        if self.args['is_train_target']:
            if self.args['is_dp_defense']:
                self.target_train_precision, self.target_test_precision, self.epsilon = self.target_model.train_model(train_dset, test_dset)
            else:
                self.target_train_precision, self.target_test_precision = self.target_model.train_model(train_dset, test_dset)

            self.data_store.save_target_model(self.target_model)
        else:
            self.data_store.load_target_model(self.target_model)
            self.target_test_precision = self.target_model.evaluate_model(test_dset)

    def train_shadow_model(self):
        self.logger.info('training shadow model')
        train_dset = TaskDataset(self.shadow_train_mem_dset, task_transforms=self.train_transforms)
        test_dset = TaskDataset(self.shadow_test_dset, task_transforms=self.test_transforms, num_tasks=self.args['test_num_task'])

        if self.args['is_train_target']:
            if self.args['is_dp_defense']:
                self.shadow_train_precision, self.shadow_test_precision, self.shadow_epsilon = self.shadow_model.train_model(train_dset, test_dset)
            else:
                self.shadow_train_precision, self.shadow_test_precision = self.shadow_model.train_model(train_dset, test_dset)
            self.data_store.save_shadow_model(self.shadow_model)
        else:
            self.data_store.load_shadow_model(self.shadow_model)

    def probe_target_model(self):
        self.logger.info('probing target model')

        if self.args['is_generate_probe']:
            if not self.args['is_sort_query']:
                train_probe_dset = ProbeDataset(self.target_train_nonmem_dset, self.target_test_dset, self.args['probe_num_task'],
                                                self.args['way'], self.args['shot'] + self.args['probe_num_query'])
                test_probe_dset = ProbeDataset(self.target_test_dset, self.target_test_dset, self.args['probe_num_task'],
                                                self.args['way'], self.args['shot'] + self.args['probe_num_query']) 
            else:
                train_probe_dset = ProbeDatasetSort(self.target_train_nonmem_dset, self.target_test_dset, self.args)
                test_probe_dset = ProbeDatasetSort(self.target_test_dset, self.target_test_dset, self.args)


            if self.args['is_known_embedding']:
                train_embeddings, train_classes = self.target_model.generate_embeddings(train_probe_dset)
                test_embeddings, test_classes = self.target_model.generate_embeddings(test_probe_dset)
                train_score, test_score = self._PuCu_calculation(train_embeddings, test_embeddings)
            elif self.args['is_memguard_defense']:
                train_score, train_classes, train_basic_similarity = self.target_model.probe_model_full(train_probe_dset)
                test_score, test_classes, test_basic_similarity = self.target_model.probe_model_full(test_probe_dset)
            else:
                train_score, train_classes, train_basic_similarity = self.target_model.probe_model(train_probe_dset)
                test_score, test_classes, test_basic_similarity = self.target_model.probe_model(test_probe_dset)

            case = "iterative"
            if self.args['is_memguard_defense']:
                if case == "random":
                    train_score = self._memguard_random_perturbe_scores(train_score)
                    test_score = self._memguard_random_perturbe_scores(test_score)
                elif case == "iterative":
                    trains, tests = [],[]
                    for _, (train, test) in enumerate(zip(train_score, test_score)):
                        trains.append(np.concatenate(self._memguard_iterative_perturbe_scores(train, self.args['epsilon'])[:self.args['probe_num_query']]))
                        tests.append(np.concatenate(self._memguard_iterative_perturbe_scores(test, self.args['epsilon'])[:self.args['probe_num_query']]))
                    train_score = np.stack(trains)
                    test_score = np.stack(tests)

            if self.args['is_noise_defense']:
                train_score = self._perturbe_scores(train_score)
                test_score = self._perturbe_scores(test_score)

            self.attack_test_data = np.concatenate((train_score, test_score))
            self.attack_test_label = np.concatenate((np.ones(train_score.shape[0]), np.zeros(test_score.shape[0])))
            self.data_store.save_attack_test_data((self.attack_test_data, self.attack_test_label))

            if self.args['is_use_image_similarity']:
                self.attack_test_data_basic = np.concatenate((train_basic_similarity, test_basic_similarity), axis=0)
                self.attack_test_data = np.concatenate((self.attack_test_data, self.attack_test_data_basic), axis=1)
                self.data_store.save_attack_test_data((self.attack_test_data, self.attack_test_label, self.attack_test_classes))
        else:
            self.attack_test_data, self.attack_test_label = self.data_store.load_attack_test_data()


    def probe_shadow_model(self):
        self.logger.info('probing shadow model')

        if self.args['is_generate_probe']:
            if not self.args['is_sort_query']:
                train_probe_dset = ProbeDataset(self.shadow_train_nonmem_dset, self.shadow_test_dset, self.args['probe_num_task'],
                                                self.args['way'], self.args['shot'] + self.args['probe_num_query'])
                test_probe_dset = ProbeDataset(self.shadow_test_dset, self.shadow_test_dset, self.args['probe_num_task'],
                                                self.args['way'], self.args['shot'] + self.args['probe_num_query'])
            else:
                train_probe_dset = ProbeDatasetSort(self.target_train_nonmem_dset, self.target_test_dset, self.args)
                test_probe_dset = ProbeDatasetSort(self.target_test_dset, self.target_test_dset, self.args)

            if self.args['is_known_embedding']:
                train_embeddings, train_classes = self.shadow_model.generate_embeddings(train_probe_dset)
                test_embeddings, test_classes = self.shadow_model.generate_embeddings(test_probe_dset)
                train_score, test_score = self._PuCu_calculation(train_embeddings, test_embeddings)
            else:
                train_score, train_classes, train_basic_similarity = self.shadow_model.probe_model(train_probe_dset)
                test_score, test_classes, test_basic_similarity = self.shadow_model.probe_model(test_probe_dset)

            self.attack_train_data = np.concatenate((train_score, test_score))
            self.attack_train_label = np.concatenate((np.ones(train_score.shape[0]), np.zeros(test_score.shape[0])))
            self.data_store.save_attack_train_data(
                (self.attack_train_data, self.attack_train_label))
            if self.args['is_use_image_similarity']:
                self.attack_train_data_basic = np.concatenate((train_basic_similarity, test_basic_similarity), axis=0)
                self.attack_train_data = np.concatenate((self.attack_train_data, self.attack_train_data_basic), axis=1)
                self.data_store.save_attack_train_data((self.attack_train_data, self.attack_train_label, self.attack_train_classes))
        else:
            self.attack_train_data, self.attack_train_label = self.data_store.load_attack_train_data()

    def save_results(self, upload_data):
        if self.args['is_use_image_similarity']:
            upload_data['probe_image_similarity'] = np.mean(self.attack_train_data[:, -(self.args['shot']):])
        else:
            upload_data['probe_image_similarity'] = 0

        if self.args['is_sort_query']:
            upload_data['image_similarity_name'] = self.args['image_similarity_name']

        if self.args['is_train_target']:
            upload_data['target_train_acc'] = self.target_train_precision
            upload_data['target_test_acc'] = self.target_test_precision
            upload_data['target_overfitting'] = self.target_train_precision - self.target_test_precision
        if self.args['is_dp_defense']:
            upload_data['epsilon'] = self.epsilon
        if self.args['is_noise_defense']:
            upload_data['noise_std'] = self.args['noise_std']
            upload_data['target_test_acc'] = self.target_test_precision

        upload_data['is_memguard_defense'] = self.args['is_memguard_defense']

        upload_data['attack_acc'] = self.attack_acc_avg
        upload_data['attack_auc'] = self.attack_auc_avg
        upload_data['attack_precision'] = self.attack_precision_avg
        upload_data['attack_recall'] = self.attack_recall_avg
        upload_data['attack_f1_score'] = self.attack_f1_score_avg
        upload_data['attack_acc_std'] = self.attack_acc_std
        upload_data['attack_auc_std'] = self.attack_auc_std
        upload_data['attack_precision_std'] = self.attack_precision_std
        upload_data['attack_recall_std'] = self.attack_recall_std
        upload_data['attack_f1_score_std'] = self.attack_f1_score_std
        upload_data['attack_class_acc'] = self.attack_acc_avg
        upload_data['attack_class_acc_std'] = self.attack_acc_std
        upload_data['attack_FPR'] = self.attack_FPR_avg
        upload_data['attack_FPR_std'] = self.attack_FPR_std
        self.write_results(upload_data)
