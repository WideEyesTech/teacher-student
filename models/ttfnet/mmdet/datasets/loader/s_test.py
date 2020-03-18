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
        self.epoch = 2
        self.flag = [0]*110000
        self.flag = self.flag + [1]*10**6
        self.flag = np.array(self.flag)
        self.samples_per_gpu = 16
        self.rank = 1
        self.num_replicas = 4

        self.coco_labels = np.where(self.flag == 0)[0]
        self.weak_labels = np.where(self.flag == 1)[0]

        self.weak_labels_groups = [self.weak_labels[x:x+10**5]
                                   for x in range(0, len(self.weak_labels), 10**5)]
        self.weak_labels_groups = list(
            filter(lambda x: len(x) == 10**5, self.weak_labels_groups))

        self.epochs = list(range(0, 110, 10))

        print("Number of epochs must be less than: ",
              len(self.weak_labels_groups)+len(self.epochs))

    def test_custom_sampler(self):

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        self.coco_labels = self.coco_labels[torch.randperm(
            len(self.coco_labels), generator=g)]

        try:
            dist = self.epochs[self.epoch]
        except IndexError:
            dist = 100

        n_of_weak_labels = int(dist/100*len(self.coco_labels))

        num_coco_samples = int(
            (len(self.coco_labels) / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu)
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

        current_weak_label_group = 0 if self.epoch < 10 else self.epoch-10

        weak_labels = self.weak_labels_groups[current_weak_label_group][:n_of_weak_labels]

        assert len(self.coco_labels) + \
            len(weak_labels) == int(self.total_size/self.samples_per_gpu) * \
            self.samples_per_gpu

        coco_count = 0
        weak_count = 0
        count = 0
        indices = []
        while len(indices) != int(self.total_size/self.samples_per_gpu)*self.samples_per_gpu:
            batch = [
                *list(self.coco_labels[coco_count:coco_count+coco_samples_per_batch]),
                *list(weak_labels[weak_count:weak_count+weak_samples_per_batch])
            ]

            try:
                assert len(batch) == self.samples_per_gpu
            except AssertionError:
                indices.extend(self.coco_labels[coco_count:])
                indices.extend(weak_labels[weak_count:])
                break

            indices.extend(batch)

            count += 1
            coco_count += coco_samples_per_batch
            weak_count += weak_samples_per_batch

        assert len(indices) == self.total_size

        # Subsamples
        if not self.num_replicas == 1:
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]

        assert len(indices) == self.num_samples


        return iter(indices)


if __name__ == '__main__':
    unittest.main()
