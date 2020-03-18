from __future__ import division
import math

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / 6))
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class CustomSampler(Sampler):
    """
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = 0
        self.num_samples = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.wflag

        self.coco_labels = np.where(self.flag == 0)[0]
        self.weak_labels = np.where(self.flag == 1)[0]

        self.weak_labels_groups = [self.weak_labels[x:x+10**5]
                                   for x in range(0, len(self.weak_labels), 10**5)]
        self.weak_labels_groups = list(
            filter(lambda x: len(x) == 10**5, self.weak_labels_groups))

        self.epochs = list(range(0, 110, 10))

        print("Number of epochs must be less than: ",
              len(self.weak_labels_groups)+len(self.epochs)-1)

    def __iter__(self):

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        self.coco_labels = self.coco_labels[torch.randperm(
            len(self.coco_labels), generator=g)]

        if self.epoch == 0:
            dist = self.epochs[0]
        else:
            try:
                dist = self.epochs[self.epoch-1]
            except IndexError:
                dist = 100

        current_weak_label_group = 0 if self.epoch < 11 else self.epoch-10

        n_of_weak_labels = int(dist/100*len(self.coco_labels)) if self.epoch < 11 else len(
            self.weak_labels_groups[current_weak_label_group])
        n_of_coco_labels = len(self.coco_labels) if self.epoch < 11 else 0

        weak_labels = self.weak_labels_groups[current_weak_label_group][:n_of_weak_labels]
        coco_labels = [] if n_of_coco_labels == 0 else self.coco_labels

        num_coco_samples = int(
            int(n_of_coco_labels / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu)
        num_weak_samples = int(
            int(n_of_weak_labels / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu)

        self.num_samples = num_coco_samples + num_weak_samples

        self.total_size = self.num_samples*self.num_replicas

        coco_samples_per_batch = int(
            num_coco_samples/int(self.num_samples/self.samples_per_gpu))
        try:
            weak_samples_per_batch = int(num_weak_samples/int(
                self.num_samples/self.samples_per_gpu))
        except ZeroDivisionError:
            weak_samples_per_batch = 0
        extra = self.samples_per_gpu - \
            (coco_samples_per_batch+weak_samples_per_batch)

        coco_samples_per_batch += extra

        try:
            assert coco_samples_per_batch + \
                weak_samples_per_batch == self.samples_per_gpu
        except AssertionError:
            import pdb
            pdb.set_trace()

        coco_count = 0
        weak_count = 0

        indices = []
        while len(indices) != int(self.total_size/self.samples_per_gpu)*self.samples_per_gpu:
            batch = [
                *list(coco_labels[coco_count:coco_count +
                                  coco_samples_per_batch]),
                *list(weak_labels[weak_count:weak_count+weak_samples_per_batch])
            ]

            try:
                assert len(batch) == self.samples_per_gpu
            except AssertionError:
                indices.extend(coco_labels[coco_count:])
                indices.extend(weak_labels[weak_count:])
                break

            indices.extend(batch)

            coco_count += coco_samples_per_batch
            weak_count += weak_samples_per_batch

        assert len(indices) == int(self.total_size/self.samples_per_gpu)*self.samples_per_gpu

        # Subsamples
        if not self.num_replicas == 1:
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
