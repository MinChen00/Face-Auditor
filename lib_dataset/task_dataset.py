# cython: language_version=3
# !/usr/bin/env python3

"""
General wrapper to help create tasks.
"""

import random

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils import collate
import numpy as np

from lib_dataset.meta_dataset import MetaDataset, UnionMetaDataset, FilteredMetaDataset
from lib_metrics.image_similarity import ImageSimilarity


def fast_allocate(n):
    result = [None] * n
    for i in range(n):
        result[i] = DataDescription(i)
    return result


class DataDescription:
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/task_dataset.py)

    **Description**

    Simple class to describe the data and its transforms in a task description.

    **Arguments**

    * **index** (int) - The index of the sample in the dataset.
    """

    def __init__(self, index):
        self.index = index
        self.transforms = []


class TaskDataset(Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/task_dataset.py)

    **Description**

    Creates a set of tasks from a given Dataset.

    In addition to the Dataset, TaskDataset accepts a list of task transformations (`task_transforms`)
    which define the kind of tasks sampled from the dataset.

    The tasks are lazily sampled upon indexing (or calling the `.sample()` method), and their
    descriptions cached for later use.
    If `num_tasks` is -1, the TaskDataset will not cache task descriptions and instead continuously resample
    new ones.
    In this case, the length of the TaskDataset is set to 1.

    For more information on tasks and task descriptions, please refer to the
    documentation of task transforms.

    **Arguments**

    * **dataset** (Dataset) - Dataset of data to compute tasks.
    * **task_transforms** (list, *optional*, default=None) - List of task transformations.
    * **num_tasks** (int, *optional*, default=-1) - Number of tasks to generate.

    **Example**
    ~~~python
    dataset = l2l.data.MetaDataset(MyDataset())
    transforms = [
        l2l.data.transforms.NWays(dataset, n=5),
        l2l.data.transforms.KShots(dataset, k=1),
        l2l.data.transforms.LoadData(dataset),
    ]
    taskset = TaskDataset(dataset, transforms, num_tasks=20000)
    for task in taskset:
        X, y = task
    ~~~
    """

    def __init__(self, dataset, task_transforms=None, num_tasks=-1, task_collate=None):
        if not isinstance(dataset, MetaDataset):
            dataset = MetaDataset(dataset)
        if task_transforms is None:
            task_transforms = []
        if task_collate is None:
            task_collate = collate.default_collate
        if num_tasks < -1 or num_tasks == 0:
            raise ValueError('num_tasks needs to be -1 (infinity) or positive.')
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.task_transforms = task_transforms
        self.sampled_descriptions = {}  # Maps indices to tasks' description dict
        self.task_collate = task_collate
        self._task_id = 0

    def sample_task_description(self):
        #  Samples a new task description.
        #  list description = fast_allocate(len(self.dataset))
        description = None
        if callable(self.task_transforms):
            return self.task_transforms(description)
        for transform in self.task_transforms:
            description = transform(description)
        return description

    def get_task(self, task_description):
        # Given a task description, creates the corresponding batch of data.
        all_data = []
        for data_description in task_description:
            data = data_description.index
            for transform in data_description.transforms:
                data = transform(data)
            all_data.append(data)
        return self.task_collate(all_data)

    def sample(self):
        """
        **Description**

        Randomly samples a task from the TaskDataset.

        **Example**
        ~~~python
        X, y = taskset.sample()
        ~~~
        """
        i = random.randint(0, len(self) - 1)
        return self[i]

    def __len__(self):
        if self.num_tasks == -1:
            # Ok to return 1, since __iter__ will run forever
            # and __getitem__ will always resample.
            return 1
        return self.num_tasks

    def __getitem__(self, i):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())
        if i not in self.sampled_descriptions:
            self.sampled_descriptions[i] = self.sample_task_description()
        task_description = self.sampled_descriptions[i]
        return self.get_task(task_description)

    def __iter__(self):
        self._task_id = 0
        return self

    def __next__(self):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())

        if self._task_id < self.num_tasks:
            task = self[self._task_id]
            self._task_id += 1
            return task
        else:
            raise StopIteration

    def __add__(self, other):
        msg = 'Adding datasets not yet supported for TaskDatasets.'
        raise NotImplementedError(msg)


class ProbeDatasetSample(Dataset):
    def __init__(self, target_dataset, aux_dataset, num_tasks, way, shot):
        if not isinstance(target_dataset, MetaDataset):
            target_dataset = MetaDataset(target_dataset)
        if not isinstance(aux_dataset, MetaDataset):
            aux_dataset = MetaDataset(aux_dataset)

        self.target_dataset = target_dataset
        self.aux_dataset = aux_dataset
        self.num_tasks = num_tasks
        self.way = way
        self.shot = shot

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        # return samples of a task
        target_label = random.sample(self.target_dataset.labels, 1)[0]
        aux_label = random.sample(self.aux_dataset.labels, self.way - 1)
        # probe_dataset = UnionMetaDataset([FilteredMetaDataset(self.target_dataset, target_class),
        #                                   FilteredMetaDataset(self.aux_dataset, aux_class)])

        target_samples_indices = random.sample(self.target_dataset.labels_to_indices[target_label], self.shot-1)
        target_samples_indices.insert(0, target_samples_indices[0])
        target_samples = [self.target_dataset[idx][0] for idx in target_samples_indices]

        aux_samples_indices = []
        for label in aux_label:
            aux_samples_indices.extend(random.sample(self.aux_dataset.labels_to_indices[label], self.shot))
        aux_samples = [self.aux_dataset[idx][0] for idx in aux_samples_indices]

        data = torch.stack(target_samples + aux_samples)
        labels = torch.tensor(np.array([[i]*self.shot for i in range(0, self.way)]).flatten())
        # labels = self.aux_dataset.indices_to_labels[target_samples_indices + aux_samples_indices]
        return data, labels, target_label, 1


class ProbeDataset(Dataset):
    def __init__(self, target_dataset, aux_dataset, num_tasks, way, shot):
        if not isinstance(target_dataset, MetaDataset):
            target_dataset = MetaDataset(target_dataset)
        if not isinstance(aux_dataset, MetaDataset):
            aux_dataset = MetaDataset(aux_dataset)

        self.target_dataset = target_dataset
        self.aux_dataset = aux_dataset
        self.num_tasks = num_tasks
        self.way = way
        self.shot = shot

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        # return samples of a task
        target_label = random.sample(self.target_dataset.labels, 1)[0]
        aux_label = random.sample(self.aux_dataset.labels, self.way - 1)

        target_samples_indices = random.sample(self.target_dataset.labels_to_indices[target_label], self.shot)
        target_samples = [self.target_dataset[idx][0] for idx in target_samples_indices]

        aux_samples_indices = []
        for label in aux_label:
            aux_samples_indices.extend(random.sample(self.aux_dataset.labels_to_indices[label], self.shot))
        aux_samples = [self.aux_dataset[idx][0] for idx in aux_samples_indices]

        data = torch.stack(target_samples + aux_samples)
        labels = torch.tensor(np.array([[i]*self.shot for i in range(0, self.way)]).flatten())
        return data, labels, target_label, torch.ones(len(labels))

    def _proto_similarity(self, target_label, random_target_samples_indices):
        proto_indices = [random_target_samples_indices[0]]*5
        candidate_indices = np.setdiff1d(self.target_dataset.labels_to_indices[target_label], proto_indices)
        image_similarities={}
        for i, image1 in enumerate(candidate_indices):
            pair_distance = []
            for j, image2 in enumerate(proto_indices):
                img1 = self.target_dataset[image1][0]
                img2 = self.target_dataset[image2][0]
                pair_distance.append(self._image_level_similarity(img1, img2))
            image_similarities[image1] = np.mean(pair_distance)

        # sort the candidate by image-level similarity
        import operator
        sort_images = dict(sorted(image_similarities.items(), key=operator.itemgetter(1),reverse=True))
        query_indices = list(sort_images.keys())[-self.args['probe_num_query']:]
        probe_similarity = []
        for query_indice in query_indices:
            probe_similarity.append(image_similarities[query_indice])
        return proto_indices+query_indices, probe_similarity

    def _image_level_similarity(self, img1, img2):
        sim = self.img_sim.pair_similarity(img1, img2)
        return sim


class ProbeDatasetSort(Dataset):
    def __init__(self, target_dataset, aux_dataset, args):
        if not isinstance(target_dataset, MetaDataset):
            target_dataset = MetaDataset(target_dataset)
        if not isinstance(aux_dataset, MetaDataset):
            aux_dataset = MetaDataset(aux_dataset)
        self.args = args
        self.target_dataset = target_dataset
        self.aux_dataset = aux_dataset
        self.num_tasks = args['probe_num_task']
        self.way = args['probe_ways']
        self.shot = args['probe_shot'] + args['probe_num_query']
        self.img_sim = ImageSimilarity(args=args)

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        # return samples of a task
        target_label = random.sample(self.target_dataset.labels, 1)[0]
        aux_label = random.sample(self.aux_dataset.labels, self.way - 1)

        target_samples_indices = random.sample(self.target_dataset.labels_to_indices[target_label], self.shot)
        sort_target_samples_indices, batch_similarity = self._sort_proto_similarity(target_label, target_samples_indices)
        target_samples = [self.target_dataset[idx][0] for idx in sort_target_samples_indices]

        aux_samples_indices = []
        for label in aux_label:
            aux_samples_indices.extend(random.sample(self.aux_dataset.labels_to_indices[label], self.shot))
        aux_samples = [self.aux_dataset[idx][0] for idx in aux_samples_indices]

        data = torch.stack(target_samples + aux_samples)
        labels = torch.tensor(np.array([[i]*self.shot for i in range(0, self.way)]).flatten())
        return data, labels, target_label, np.array(batch_similarity)

    def _sort_proto_similarity(self, target_label, random_target_samples_indices):
        # proto_indices = random_target_samples_indices[0:self.args['probe_shot']]
        proto_indices = [random_target_samples_indices[0]]*5
        candidate_indices = np.setdiff1d(self.target_dataset.labels_to_indices[target_label], proto_indices)
        image_similarities={}
        for i, image1 in enumerate(candidate_indices):
            pair_distance = []
            for j, image2 in enumerate(proto_indices):
                img1 = self.target_dataset[image1][0]
                img2 = self.target_dataset[image2][0]
                pair_distance.append(self._image_level_similarity(img1, img2))
            image_similarities[image1] = np.mean(pair_distance)

        # sort the candidate by image-level similarity
        import operator
        sort_images = dict(sorted(image_similarities.items(), key=operator.itemgetter(1),reverse=True))
        query_indices = list(sort_images.keys())[-self.args['probe_num_query']:]
        probe_similarity = []
        for query_indice in query_indices:
            probe_similarity.append(image_similarities[query_indice])
        return proto_indices+query_indices, probe_similarity

    def _image_level_similarity(self, img1, img2):
        sim = self.img_sim.pair_similarity(img1, img2)
        return sim
