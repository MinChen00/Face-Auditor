import logging
from matplotlib.pyplot import axis
from scipy.special import softmax

from exp.exp_class_mem_infer import ExpClassMemInfer
from lib_dataset.meta_dataset import FilteredMetaDataset
from lib_model.siamese.siamese import Siamese
from lib_dataset.siamese_dataset import *


class ExpClassMemInferSiamese(ExpClassMemInfer):
    def __init__(self, args):
        super(ExpClassMemInferSiamese, self).__init__(args)
        self.args = args
        self.logger = logging.getLogger('exp_class_mem_infer_siamese')
        self.target_model = Siamese(self.args)
        self.shadow_model = Siamese(self.args)
        attack_acc, attack_auc, attack_precision, attack_recall, attack_f1_score, attack_class_acc, attack_FPR = [],[],[],[],[],[],[]

        for run in range(self.args['num_runs']):
            if self.args['is_train_target']:
                self.train_target_model()
            else:
                self.data_store.load_target_model(self.target_model)
                test_dset = SiameseTestDataset(self.target_test_dset, times=self.args['test_times'], way=self.args['way'])
                self.target_test_precision = self.target_model.evaluate_model(test_dset)

            if self.args['is_train_shadow']:
                self.train_shadow_model()
            else:
                self.data_store.load_shadow_model(self.shadow_model)
                test_dset = SiameseTestDataset(self.shadow_test_dset, times=self.args['test_times'], way=self.args['way'])
                self.shadow_test_precision = self.shadow_model.evaluate_model(test_dset)

            self.logger.info("**********Run %f**********" % run)
            self.probe_target_model()
            self.probe_shadow_model()
            self.basic_similarity = self.attack_train_data[:, -(self.args['shot']):]

            if self.args['attack_model'] in ['MLP', 'RF']:
                self.attack_class_acc = 0
                self.train_attack_model()
                self.evaluate_attack_model()

            attack_acc.append(self.attack_acc)
            attack_auc.append(self.attack_auc)
            attack_precision.append(self.attack_precision)
            attack_recall.append(self.attack_recall)
            attack_f1_score.append(self.attack_f1_score)
            attack_class_acc.append(self.attack_class_acc)
            attack_FPR.append(self.attack_FPR)
            # self.visualize()

        self.attack_acc_avg = np.average(attack_acc)
        self.attack_auc_avg = np.average(attack_auc)
        self.attack_precision_avg = np.average(attack_precision)
        self.attack_recall_avg = np.average(attack_recall)
        self.attack_f1_score_avg = np.average(attack_f1_score)
        self.attack_FPR_avg = np.average(attack_FPR)

        self.attack_acc_std = np.std(attack_acc)
        self.attack_auc_std = np.std(attack_auc)
        self.attack_precision_std = np.std(attack_precision)
        self.attack_recall_std = np.std(attack_recall)
        self.attack_f1_score_std = np.std(attack_f1_score)
        self.attack_class_acc_std = np.std(attack_class_acc)
        self.attack_FPR_std = np.std(attack_FPR)        
        # self.visualize()

    def train_target_model(self):
        self.logger.info('training target model')
        if self.args['is_disjoint_train']:
            train_dset = SiameseTrainDataset(self.target_train_mem_dset, self.length)
        else:
            train_dset = SiameseTrainDataset(self.target_train_dset, self.length)

        val_indices = np.random.choice(np.arange(len(self.target_train_mem_dset.labels)),
                                        len(self.target_test_dset.labels))
        val_dset = FilteredMetaDataset(self.target_train_mem_dset, val_indices)
        val_dset = SiameseTestDataset(val_dset, times=self.args['test_times'], way=self.args['way'])
        test_dset = SiameseTestDataset(self.target_test_dset, times=self.args['test_times'], way=self.args['way'])

        self.target_train_precision, self.target_test_precision = self.target_model.train_model(train_dset, val_dset, test_dset)

        self.data_store.save_target_model(self.target_model)

    def train_shadow_model(self):    
        self.logger.info('training shadow model')

        if self.args['is_disjoint_train']:
            train_dset = SiameseTrainDataset(self.shadow_train_mem_dset, self.length)
        else:
            train_dset = SiameseTrainDataset(self.shadow_train_dset, self.length)

        val_dset = SiameseTestDataset(self.shadow_train_mem_dset, times=self.args['test_times'], way=self.args['way'])
        test_dset = SiameseTestDataset(self.shadow_test_dset, times=self.args['test_times'], way=self.args['way'])

        self.shadow_train_precision, self.shadow_test_precision = self.shadow_model.train_model(train_dset, val_dset, test_dset)
        self.data_store.save_shadow_model(self.shadow_model)

    def _check_target_balanced_attack(self,):
        # balance the train_test classes
        if self.args['is_balance_attack']:
            train_probe_dset_indices = np.random.choice(np.arange(len(self.target_train_nonmem_dset.labels)),
                                                        len(self.target_test_dset.labels))
            target_train_nonmem_dset = FilteredMetaDataset(self.target_train_nonmem_dset, train_probe_dset_indices)
        else:
            target_train_nonmem_dset = self.target_train_nonmem_dset
        return target_train_nonmem_dset

    def _check_target_disjoint_train(self, target_train_nonmem_dset):
        if self.args['is_disjoint_train']:
            train_probe_dset = SiameseProbeDataset(target_train_nonmem_dset, args=self.args)
        else:
            train_probe_dset = SiameseProbeDataset(self.target_train_dset, args=self.args)
        return train_probe_dset

    def probe_target_model(self):
        self.logger.info('probing target model')
        if self.args['is_generate_probe']:
             # Not use similarity as a guidance for choosing the probe images
            if not self.args['is_similarity_aided']:
                target_train_nonmem_dset = self._check_target_balanced_attack()
                train_probe_dset = self._check_target_disjoint_train(target_train_nonmem_dset)
                test_probe_dset = SiameseProbeDataset(self.target_test_dset, args=self.args)
                train_score, train_classes, train_basic_similarity = self.target_model.probe_model(train_probe_dset)
                test_score, test_classes, test_basic_similarity = self.target_model.probe_model(test_probe_dset)
            # Use similarity as a guidance for choosing the probe images
            else:
                target_train_nonmem_dset = self._check_target_balanced_attack()

                if self.args['is_disjoint_probe']:
                    train_probe_pairs = SiameseProbePairs(self.target_train_nonmem_dset, args=self.args)
                else:
                    train_probe_pairs = SiameseProbePairs(self.target_train_dset, args=self.args)

                test_probe_pairs = SiameseProbePairs(self.target_test_dset, args=self.args)

                if not self.args['is_known_embedding']:
                    train_score, train_classes, train_basic_similarity = self.target_model.probe_model(train_probe_pairs)
                    test_score, test_classes, test_basic_similarity = self.target_model.probe_model(test_probe_pairs)
                else:
                    train_embeddings, train_classes = self.target_model.generate_embeddings(train_probe_pairs)
                    test_embeddings, test_classes = self.target_model.generate_embeddings(test_probe_pairs)
                    train_score, test_score = self._PuCu_calculation(train_embeddings, test_embeddings)

            if self.args["is_normalize_similarity"]:
                self.attack_test_data = softmax(np.concatenate((train_score, test_score)), axis=1)
            else:
                self.attack_test_data = np.concatenate((train_score, test_score))

            if self.args['is_noise_defense']:
                train_score = self._perturbe_scores(train_score)
                test_score = self._perturbe_scores(test_score)

            if self.args['is_memguard_defense']:
                train_score = self._memguard_perturbe_scores(train_score)
                test_score = self._memguard_perturbe_scores(test_score)

            self.attack_test_label = np.concatenate((np.ones(train_score.shape[0]), np.zeros(test_score.shape[0])))
            self.attack_test_classes = np.concatenate((train_classes, test_classes))
            self.data_store.save_attack_test_data((self.attack_test_data, self.attack_test_label, self.attack_test_classes))

            if self.args['is_use_image_similarity']:
                self.attack_test_data_basic = np.concatenate((train_basic_similarity, test_basic_similarity), axis=0)
                self.attack_test_data = np.concatenate((softmax(self.attack_test_data, axis=1), self.attack_test_data_basic), axis=1)
                self.data_store.save_attack_test_data((self.attack_test_data, self.attack_test_label, self.attack_test_classes))

        else:
            self.attack_test_data, self.attack_test_label, self.attack_test_classes = self.data_store.load_attack_test_data()

    def probe_shadow_model(self):
        self.logger.info('probing shadow model')
        if self.args['is_generate_probe']:
             # Not use similarity as a guidance for choosing the probe images
            if not self.args['is_similarity_aided']:
                if self.args['is_disjoint_train']:
                    train_probe_dset = SiameseProbeDataset(self.shadow_train_nonmem_dset, args=self.args)
                else:
                    train_probe_dset = SiameseProbeDataset(self.shadow_train_dset, args=self.args)
                test_probe_dset = SiameseProbeDataset(self.shadow_test_dset, args=self.args)

                train_score, train_classes, train_basic_similarity = self.shadow_model.probe_model(train_probe_dset)
                test_score, test_classes, test_basic_similarity = self.shadow_model.probe_model(test_probe_dset)
            
            # Use similarity as a guidance for choosing the probe images
            else:
                # choose args['shot'] image pairs from each class, repeat args['prob_times'] times
                # in total, we get args['prob_times'] args['shot']-dimensional pairs for calculating similarity score.
                if self.args['is_disjoint_probe']:
                    train_probe_pairs = SiameseProbePairs(self.shadow_train_nonmem_dset, args=self.args)
                else:
                    train_probe_pairs = SiameseProbePairs(self.shadow_train_dset, args=self.args)

                test_probe_pairs = SiameseProbePairs(self.shadow_test_dset, args=self.args)

                if not self.args['is_known_embedding']:
                    train_score, train_classes, train_basic_similarity = self.shadow_model.probe_model(train_probe_pairs)
                    test_score, test_classes, test_basic_similarity = self.shadow_model.probe_model(test_probe_pairs)
                else:
                    train_embeddings, train_class = self.shadow_model.generate_embeddings(train_probe_pairs)
                    test_embeddings, train_class = self.shadow_model.generate_embeddings(test_probe_pairs)
                    train_score, test_score = self._PuCu_calculation(train_embeddings, test_embeddings)

            if self.args["is_normalize_similarity"]:
                self.attack_train_data = softmax(np.concatenate((train_score, test_score)), axis=1)
            else:
                self.attack_train_data = np.concatenate((train_score, test_score))

            self.attack_train_label = np.concatenate((np.ones(train_score.shape[0]), np.zeros(test_score.shape[0])), axis=0)
            self.attack_train_classes = np.concatenate((train_classes, test_classes))
            self.data_store.save_attack_train_data((self.attack_train_data, self.attack_train_label, self.attack_train_classes))

            if self.args['is_use_image_similarity']:
                self.attack_train_data_basic = np.concatenate((train_basic_similarity, test_basic_similarity), axis=0)
                self.attack_train_data = np.concatenate((softmax(self.attack_train_data, axis=1), self.attack_train_data_basic), axis=1)
                self.data_store.save_attack_train_data((self.attack_train_data, self.attack_train_label, self.attack_train_classes))
        else:
            self.attack_train_data, self.attack_train_label, self.attack_train_classes = self.data_store.load_attack_train_data()

    def save_results(self, upload_data):
        if self.args['is_train_target']:
            upload_data['target_train_acc'] = self.target_train_precision
            upload_data['target_test_acc'] = self.target_test_precision
            upload_data['target_overfitting'] = self.target_train_precision - self.target_test_precision

        if self.args['is_noise_defense']:
            upload_data['target_test_acc'] = self.target_test_precision

        if self.args['is_use_image_similarity']:
            upload_data['probe_image_similarity'] = np.mean(self.attack_train_data[:, -(self.args['shot']):])
        else:
            upload_data['probe_image_similarity'] = 0

        if self.args['is_dp_defense']:
            upload_data['epsilon'] = self.epsilon

        upload_data['image_similarity_name'] = self.args['image_similarity_name']
        upload_data['attack_class_acc'] = self.attack_class_acc
        upload_data['attack_class_acc_std'] = self.attack_class_acc_std
        upload_data['attack_acc'] = self.attack_acc
        upload_data['attack_acc_std'] = self.attack_acc_std
        upload_data['attack_auc'] = self.attack_auc
        upload_data['attack_auc_std'] = self.attack_auc_std
        upload_data['attack_precision'] = self.attack_precision
        upload_data['attack_precision_std'] = self.attack_precision_std
        upload_data['attack_recall'] = self.attack_recall
        upload_data['attack_recall_std'] = self.attack_recall_std
        upload_data['attack_f1_score'] = self.attack_f1_score
        upload_data['attack_f1_score_std'] = self.attack_f1_score_std
        upload_data['attack_FPR'] = self.attack_FPR_avg
        upload_data['attack_FPR_std'] = self.attack_FPR_std

        if self.args['is_save_results']:
            self.write_results(upload_data)
