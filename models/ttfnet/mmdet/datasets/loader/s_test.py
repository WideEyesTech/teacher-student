import unittest
import numpy as np
import torch
import math
from random import shuffle


class Test(unittest.TestCase):
    '''
    '''

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.epoch = 0
        self.flag = [0]*117000
        self.flag = self.flag + [1]*10**6
        self.flag = np.array(self.flag)
        self.samples_per_gpu = 16
        self.rank = 1
        self.num_replicas = 1

        self.coco_labels = np.where(self.flag == 0)[0]
        self.weak_labels = np.where(self.flag == 1)[0]

        self.weak_labels_groups = [self.weak_labels[x:x+10**5]
                                   for x in range(0, len(self.weak_labels), 10**5)]
        self.weak_labels_groups = list(
            filter(lambda x: len(x) == 10**5, self.weak_labels_groups))

        self.epochs = list(range(0, 110, 10))

        print("Number of epochs must be less than: ",
              len(self.weak_labels_groups)+len(self.epochs)-1)

    def test_custom_sampler(self):

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

        n_of_weak_labels = int(dist/100*len(self.coco_labels)) if self.epoch < 11 else len(self.weak_labels_groups[current_weak_label_group])
        n_of_coco_labels = len(self.coco_labels) if self.epoch < 11 else 0


        weak_labels = self.weak_labels_groups[current_weak_label_group][:n_of_weak_labels]
        coco_labels = [] if n_of_coco_labels == 0 else self.coco_labels

        num_coco_samples = int(
            (n_of_coco_labels / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu)
        num_weak_samples = int(
            (n_of_weak_labels / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu)

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
        
        coco_samples_per_batch+=extra

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
                *list(coco_labels[coco_count:coco_count+coco_samples_per_batch]),
                *list(weak_labels[weak_count:weak_count+weak_samples_per_batch])
            ]

            if len(batch) != self.samples_per_gpu:
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


if __name__ == '__main__':
    unittest.main()
