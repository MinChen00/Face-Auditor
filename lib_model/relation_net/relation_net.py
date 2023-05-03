import logging
import os

from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from opacus import PrivacyEngine
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector

from lib_dataset.data_store import DataStore
from lib_model.relation_net.model import CNNEncoder, FeatureEncoder, RelationNetwork
from lib_model.few_shot_base import FewShotBase
from parameter_parser import parameter_parser
from lib_dataset.task_dataset import TaskDataset
from lib_dataset.transform import NWays, KShots, LoadData, RemapLabels


class RelationNet(FewShotBase):
    def __init__(self, args):
        super(RelationNet, self).__init__(args)
        self.logger = logging.getLogger('relation_net')
        # self.feat_dim = 64  # default image_size: 32:64
        input_size = {
            32: 64,
            84: 64,
            96: 64,
            112: 64,
            160: 64,
            224: 64
        }
        self.feat_dim = input_size[self.args['image_size']]
        self.relation_dim = 8

        if self.args['dataset_name'] in ['vggface2', 'webface', 'umdfaces', 'celeba']:
            self.feat_encoder = FeatureEncoder(self.args['feature_extractor'], in_channels=3).to(self.device)
        else:
            raise Exception("Unsupported dataset.")
        self.model = RelationNetwork(self.feat_dim, self.relation_dim).to(self.device)
        if self.args['is_dp_defense']:
            self.model = self.convert_layers(self.model, nn.BatchNorm2d, nn.GroupNorm, True, num_groups=2).to(self.device) # for DP
            # self.model = module_modification.convert_batchnorm_modules(self.model).to(self.device)

    def convert_layers(self, model, layer_type_old, layer_type_new, convert_weights=False, num_groups=None):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                # Recursion.
                model._modules[name] = self.convert_layers(module, layer_type_old, layer_type_new, convert_weights)
            if type(module) == layer_type_old:
                layer_old = module
                # If num_groups is None, GroupNorm turns into InstanceNorm
                # If num_groups is 1, GroupNorm turns into LayerNorm. 
                layer_new = layer_type_new(module.num_features if num_groups is None else num_groups, module.num_features, module.eps, module.affine) 
                if convert_weights:
                    layer_new.weight = layer_old.weight
                    layer_new.bias = layer_old.bias
                model._modules[name] = layer_new
        return model


    def train_model(self, train_dset, test_dset):
        self.model.train()
        train_loader = DataLoader(train_dset, pin_memory=True, shuffle=True)

        feat_encoder_optim = torch.optim.Adam(self.feat_encoder.parameters(), lr=0.001)
        feat_encoder_scheduler = StepLR(feat_encoder_optim, step_size=10, gamma=0.5)
        relation_net_optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        relation_net_scheduler = StepLR(relation_net_optim, step_size=10, gamma=0.5)

        if self.args['is_dp_defense']:
            privacy_engine = PrivacyEngine(
                self.model,
                sample_rate=self.args['sample_rate'],
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args['sigma'],
                max_grad_norm=self.args['max_per_sample_grad_norm'],
                secure_rng=self.args['secure_rng'],
            )
            privacy_engine.attach(relation_net_optim)

        for epoch in range(self.args['train_num_epochs']):
            self.model.train()
            self.logger.info('epoch: %s' % (epoch,))

            for i in range(self.args['train_num_task']):
                data, labels = next(iter(train_loader))

                _, loss, _ = self.fast_adapt(data, labels, self.args['way'], self.args['shot'],
                                             self.args['train_num_query'])

                # training
                self.feat_encoder.zero_grad()
                self.model.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.feat_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                feat_encoder_optim.step()
                relation_net_optim.step()

                if self.args['is_dp_defense']:
                    epsilon, best_alpha = relation_net_optim.privacy_engine.get_privacy_spent(self.args['delta'])
                    self.logger.info("Epoch: %s | epsilon: %s | best alpha: %d" % (epoch, epsilon, best_alpha))

            feat_encoder_scheduler.step(epoch)
            relation_net_scheduler.step(epoch)

            train_acc = self.evaluate_model(train_dset)
            test_acc = self.evaluate_model(test_dset)
            self.logger.debug('train acc: %s | test acc: %s' % (train_acc, test_acc,))

        if self.args['is_dp_defense']:
            return train_acc, test_acc, epsilon
        else:
            return train_acc, test_acc

    def evaluate_model(self, test_dset):
        self.model.eval()

        test_loader = DataLoader(test_dset, pin_memory=True, shuffle=True)

        loss_ctr = 0
        n_acc = 0
        for i, (data, labels) in enumerate(test_loader):
            _, _, acc = self.fast_adapt(data, labels, self.args['way'], self.args['shot'], self.args['test_num_query'])
            loss_ctr += 1
            n_acc += acc

        return (n_acc / loss_ctr).item()

    def probe_model(self, probe_dset):
        self.model.eval()
        probe_loader = DataLoader(probe_dset, pin_memory=True, shuffle=True)
        ret_logit = []
        train_class = []
        batch_similarity = []

        for i, (data, labels, class_label, similarity) in enumerate(probe_loader, 1):
            self.logger.info("batch %d" % i)
            relations, _, _ = self.fast_adapt(data, labels, self.args['probe_ways'], self.args['probe_shot'],
                                              self.args['probe_num_query'])

            # 1. concatenate all query samples with target label
            ret_logit.append(np.concatenate(relations[:self.args['probe_num_query']].detach().cpu().numpy()))
            train_class.append(class_label)
            batch_similarity.append(similarity)

        return np.stack(ret_logit), torch.stack(train_class).numpy(), torch.cat(batch_similarity).numpy()

    def probe_model_full(self, probe_dset):
        self.model.eval()
        self.model.to(self.device)

        probe_loader = DataLoader(probe_dset, pin_memory=True, shuffle=True)
        ret_logit = []
        train_class = []
        batch_similarity = []

        for i, (data, labels, class_label, similarity) in enumerate(probe_loader, 1):
            relations, _, _ = self.fast_adapt(data, labels, self.args['probe_ways'], self.args['probe_shot'],
                                              self.args['probe_num_query'])
            # 4. concatenate all query samples, including the auxiliary query samples
            ret_logit.append(relations.detach().cpu().numpy())
            train_class.append(class_label)
            batch_similarity.append(similarity)    
        return np.stack(ret_logit), torch.stack(train_class).numpy(), torch.cat(batch_similarity).numpy()

    def fast_adapt(self, data, labels, ways, shot, query_num):
        data = data.to(self.device)
        labels = labels.to(self.device)

        # Sort data samples by labels
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + query_num)
        for offset in range(shot):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        samples = data[support_indices]
        batches = data[query_indices]
        batch_labels = labels[query_indices]

        # calculate features
        sample_features = self.feat_encoder(samples)  # 5x64*5*5
        sample_features = sample_features.view(ways, shot, self.feat_dim,
                                               sample_features.shape[2],
                                               sample_features.shape[3])
        sample_features = torch.mean(sample_features, 1).squeeze(1)
        batch_features = self.feat_encoder(batches)  # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(query_num * ways, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(ways, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        # 5, 5 should be the same size of sample features
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1,
                                                                                      self.feat_dim * 2,
                                                                                      sample_features_ext.shape[3],
                                                                                      sample_features_ext.shape[4])
        relations = self.model(relation_pairs).view(-1, ways)

        mse = nn.MSELoss()
        loss = mse(relations, F.one_hot(batch_labels).float())
        acc = self._accuracy(relations, batch_labels)

        perturb_logits = self._perturbe_scores(relations)
        perturb_acc = self._accuracy(perturb_logits, batch_labels)
        print(str(acc), str(perturb_acc))
        return relations, loss, acc

    def get_embeddings(self, data, labels, ways, shot, query_num):
        data = data.to(self.device)
        labels = labels.to(self.device)

        # Sort data samples by labels
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + query_num)
        for offset in range(shot):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        samples = data[support_indices]
        batches = data[query_indices]
        batch_labels = labels[query_indices]

        # calculate features
        sample_features = self.feat_encoder(samples)  # 5x64*5*5
        sample_features = sample_features.view(ways, shot, self.feat_dim,
                                               sample_features.shape[2],
                                               sample_features.shape[3])
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        batch_features = self.feat_encoder(batches)  # 20x64*5*5
        return batch_features

    def generate_embeddings(self, probe_dset):
        self.model.eval()
        self.model.to(self.device)

        probe_loader = DataLoader(probe_dset, pin_memory=True, shuffle=True)
        ret_embeddings = []
        train_class = []

        for i, (data, labels, class_label, similarity) in enumerate(probe_loader, 1):
            embeddings = self.get_embeddings(data, labels, self.args['probe_ways'], self.args['probe_shot'], self.args['probe_num_query'])
            ret_embeddings.append(np.concatenate(embeddings[:self.args['probe_num_query']].detach().cpu().numpy()))
            train_class.append(class_label)
        return np.stack(ret_embeddings), train_class

    def _perturbe_scores(self, scores):
        scores=scores.cpu().detach().numpy()
        ret_scores = np.zeros_like(scores)
        for i, score in enumerate(scores):
            ret_scores[i] = score + np.random.laplace(loc=0.0, scale=self.args['noise_std'], size=score.size)
        return torch.from_numpy(ret_scores).to(self.device)

    def _accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        acc = (predictions == targets).sum().float() / targets.size(0)
        return acc

    def probe_adapt(self):
        pass



if __name__ == '__main__':
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    args = parameter_parser()
    relationnet = RelationNet(args)

    data_store = DataStore(args)

    target_train_dset, shadow_train_dset, \
    target_train_mem_dset, target_train_nonmem_dset, \
    shadow_train_mem_dset, shadow_train_nonmem_dset, \
    target_test_dset, shadow_test_dset = \
        data_store.load_data()
    train_transforms = [
        NWays(target_train_mem_dset, args['way']),
        KShots(target_train_mem_dset, args['train_num_query'] + args['shot']),
        LoadData(target_train_mem_dset),
        RemapLabels(target_train_mem_dset),
    ]
    test_transforms = [
        NWays(target_test_dset, args['way']),
        KShots(target_test_dset, args['test_num_query'] + args['shot']),
        LoadData(target_test_dset),
        RemapLabels(target_test_dset),
    ]

    train_dset = TaskDataset(target_train_mem_dset, task_transforms=train_transforms)
    test_dset = TaskDataset(target_test_dset, task_transforms=test_transforms,
                            num_tasks=args['test_num_task'])
    relationnet.train_model(train_dset, test_dset)
