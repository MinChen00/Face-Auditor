import logging
from re import S

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from opacus import PrivacyEngine
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification

from lib_dataset.data_store import DataStore
from lib_model.proto_net.model import Convnet, PrototypicalNetwork
from lib_model.few_shot_base import FewShotBase
from parameter_parser import parameter_parser
from lib_dataset.task_dataset import TaskDataset
from lib_dataset.transform import NWays, KShots, LoadData, RemapLabels


class ProtoNet(FewShotBase):
    def __init__(self, args):
        super(ProtoNet, self).__init__(args)
        self.logger = logging.getLogger('proto_net')

        self.model = PrototypicalNetwork(self.args['feature_extractor'], 3, 64, 64).to(self.device)
        # if self.args['is_dp_defense']:
        #     self.model = self.convert_layers(self.model, nn.BatchNorm2d, nn.GroupNorm, True, num_groups=2) # for DP
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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        if self.args['is_dp_defense']:
            privacy_engine = PrivacyEngine(
                self.model,
                sample_rate=self.args['sample_rate'],
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args['sigma'],
                max_grad_norm=self.args['max_per_sample_grad_norm'],
                secure_rng=self.args['secure_rng'],
            )
            privacy_engine.attach(optimizer)

        for epoch in range(self.args['train_num_epochs']):
            self.logger.info('epoch: %s' % (epoch,))
            self.model.train()

            for i in range(self.args['train_num_task']):
                data, labels = next(iter(train_loader))
                _, loss, _ = self.fast_adapt(data, labels, self.args['way'], self.args['shot'], self.args['train_num_query'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args['is_dp_defense']:
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.args['delta'])
                    self.logger.info("Epoch: %s | epsilon: %s | best alpha: %s" % (epoch, epsilon, best_alpha))

            lr_scheduler.step()

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
        for i in range(self.args['test_num_task']):
            data, labels = next(iter(test_loader))
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
            logit, _, _ = self.fast_adapt(data, labels,
                                          self.args['probe_ways'], self.args['probe_shot'],
                                          self.args['probe_num_query'])

            # 1. concatenate all query samples with target label
            ret_logit.append(np.concatenate(logit[:self.args['probe_num_query']].detach().cpu().numpy()))
            # 2. only use the the last query sample
            # ret_logit.append(logit[self.args['probe_num_query']].detach().cpu().numpy())
            # 3. only concatenate all the target logits bit
            # ret_logit.append(np.concatenate(logit[:self.args['probe_num_query'], 0].detach().cpu().numpy().reshape(1,-1)))
            # 4. concatenate all query samples, including the auxilary query samples
            # ret_logit.append(np.concatenate(logit.detach().cpu().numpy()))
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
            logit, _, _ = self.fast_adapt(data, labels,
                                          self.args['probe_ways'], self.args['probe_shot'],
                                          self.args['probe_num_query'])
            # 4. concatenate all query samples, including the auxilary query samples
            ret_logit.append(logit.detach().cpu().numpy())
            train_class.append(class_label)
            batch_similarity.append(similarity)

        return np.stack(ret_logit), torch.stack(train_class).numpy(), torch.cat(batch_similarity).numpy()

    def fast_adapt(self, data, labels, ways, shot, query_num):
        data = data.to(self.device)
        labels = labels.to(self.device)
        n_items = shot * ways

        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        # Compute support and query embeddings
        embeddings = self.model(data)
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(ways) * (shot + query_num)
        for offset in range(shot):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = embeddings[support_indices]
        support = support.reshape(ways, shot, -1).mean(dim=1)
        query = embeddings[query_indices]
        labels = labels[query_indices].long()

        logits = self._pairwise_distances_logits(query, support)
        loss = F.cross_entropy(logits, labels)
        acc = self._accuracy(logits, labels)
        
        perturb_logits = self._perturbe_scores(logits)
        perturb_acc = self._accuracy(perturb_logits, labels)
        print(str(acc), str(perturb_acc))

        return logits, loss, acc

    def generate_embeddings(self, probe_dset):
        self.model.eval()
        self.model.to(self.device)

        train_class,embedding = [],[]

        probe_loader = DataLoader(probe_dset, pin_memory=False, shuffle=True)
        for i, (data, labels, class_label, similarity) in enumerate(probe_loader, 1):
            data = data.to(self.device)
            labels = labels.to(self.device)

            # Sort data samples by labels
            sort = torch.sort(labels)
            data = data.squeeze(0)[sort.indices].squeeze(0)
            labels = labels.squeeze(0)[sort.indices].squeeze(0)
            train_class.append(class_label)
            with torch.no_grad():
                data_embedding=self.model(data).cpu()
                embedding.append(data_embedding)

        # Compute support and query embeddings
        embeddings = np.stack(embedding)
        return embeddings, train_class

    def _perturbe_scores(self, scores):
        scores=scores.cpu().detach().numpy()
        ret_scores = np.zeros_like(scores)
        for i, score in enumerate(scores):
            ret_scores[i] = score + np.random.laplace(loc=0.0, scale=self.args['noise_std'], size=score.size)
        return torch.from_numpy(ret_scores).to(self.device)
    
    def probe_adapt(self):
        pass

    def _pairwise_distances_logits(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        logits = -((a.unsqueeze(1).expand(n, m, -1) -
                    b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
        return logits

    def _accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)


if __name__ == '__main__':
    import os
    os.chdir('../../')
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    args = parameter_parser()
    protonet = ProtoNet(args)

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
    protonet.train_model(train_dset, test_dset)
