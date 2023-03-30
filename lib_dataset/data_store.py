import pickle

import config


class DataStore:
    def __init__(self, args):
        self.args = args

        self.determine_data_path()
        self.generate_folder()

    def determine_data_path(self):
        target_model_name = '_'.join(('target', str(self.args['dataset_task']), self.args['dataset_name'], self.args['feature_extractor'],
                                      str(self.args['batch_size']), str(self.args['num_iter']),
                                      str(self.args['way']), str(self.args['shot']), str(self.args['image_size'])))
        shadow_model_name = '_'.join(('shadow', str(self.args['dataset_task']), self.args['shadow_dataset_name'], self.args['feature_extractor'],
                                      str(self.args['batch_size']), str(self.args['num_iter']),
                                      str(self.args['way']), str(self.args['shot']), str(self.args['image_size'])))
        attack_train_data = '_'.join((self.args['dataset_name'], self.args["shadow_model"], str(self.args['probe_shot']), "attack_train_data"))
        attack_test_data = '_'.join((self.args['dataset_name'], self.args["target_model"], str(self.args['probe_shot']), "attack_test_data"))

        if self.args['is_disjoint_train']:
            target_model_name += "_disjoint"
            shadow_model_name += "_disjoint"

        if self.args['is_use_image_similarity']:
            attack_train_data = '_'.join((attack_train_data, "base",
                                          str(self.args['image_similarity_level']),
                                          str(self.args['image_similarity_name'])))
            attack_test_data = '_'.join((attack_test_data, "base",
                                        str(self.args['image_similarity_level']),
                                        str(self.args['image_similarity_name'])))
        else:
            attack_train_data = '_'.join((attack_train_data, "clean", str(self.args['image_similarity_name'])))
            attack_test_data = '_'.join((attack_test_data, "clean", str(self.args['image_similarity_name'])))
        
        if self.args['is_dp_defense']:
            target_model_name = "_".join((target_model_name, "DP"))
            shadow_model_name = "_".join((shadow_model_name, "DP"))
            attack_train_data = "_".join((attack_train_data, "DP"))
            attack_test_data = "_".join((attack_test_data, "DP"))

        if self.args['is_adv_defense']:
            target_model_name = "_".join((target_model_name, "ADV", self.args['fawkes_mode']))
            shadow_model_name = "_".join((shadow_model_name, "ADV", self.args['fawkes_mode']))
            attack_train_data = "_".join((attack_train_data, "ADV", self.args['fawkes_mode']))
            attack_test_data = "_".join((attack_test_data, "ADV", self.args['fawkes_mode']))

        if self.args['is_noise_defense']:
            attack_train_data = "_".join((attack_train_data, "Noise"))
            attack_test_data = "_".join((attack_test_data, "Noise"))

        self.attack_train_data = config.ATTACK_DATA_PATH + self.args['dataset_name'] + "/" + attack_train_data
        self.attack_test_data = config.ATTACK_DATA_PATH + self.args['dataset_name'] + "/" + attack_test_data
        self.target_model_file = config.MODEL_PATH + self.args['target_model'] + "/" + target_model_name
        self.shadow_model_file = config.MODEL_PATH + self.args['target_model'] + "/" + shadow_model_name

    def generate_folder(self):
        pass

    def load_data(self):
        data_path = config.PROCESSED_DATA_PATH + str(self.args['image_size']) + "/" + "_".join((str(self.args['dataset_task']), self.args['dataset_name']))
        if self.args['is_adv_defense']:
            data_path += "_" + self.args['fawkes_mode']
        print(data_path)
        return pickle.load(open(data_path, 'rb'))

    def save_target_model(self, target_model):
        target_model.save_model(self.target_model_file)

    def load_target_model(self, target_model):
        target_model.load_model(self.target_model_file)

    def save_shadow_model(self, shadow_model):
        shadow_model.save_model(self.shadow_model_file)

    def load_shadow_model(self, shadow_model):
        shadow_model.load_model(self.shadow_model_file)

    def save_attack_train_data(self, attack_train_data):
        pickle.dump((attack_train_data), open(self.attack_train_data, 'wb'))

    def load_attack_train_data(self):
        attack_train_data = pickle.load(open(self.attack_train_data, 'rb'))
        return attack_train_data[0], attack_train_data[1]

    def save_attack_test_data(self, attack_test_data):
        pickle.dump((attack_test_data), open(self.attack_test_data, 'wb'))

    def load_attack_test_data(self):
        attack_test_data = pickle.load(open(self.attack_test_data, 'rb'))
        return attack_test_data[0], attack_test_data[1]
