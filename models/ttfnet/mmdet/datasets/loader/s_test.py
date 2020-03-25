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
        self.epoch = 1
        self.flag = [0]*120506
        self.flag = self.flag + [1]*10**6
        self.flag = np.array(self.flag)
        self.samples_per_gpu = 32
        self.rank = 1
        self.num_replicas = 1

        self.coco_labels = np.where(self.flag == 0)[0]
        self.weak_labels = np.where(self.flag == 1)[0]

        self.weak_labels_group_size = int(len(self.weak_labels)/10**5)
        self.group_iter = 0
        print("Weak labels groups size: ", self.weak_labels_group_size)
        print("Group iter: ", self.group_iter)

        self.epochs = list(range(0, 110, 10))


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
                dist = self.epochs[self.epoch]
            except IndexError:
                dist = 100
        n_of_weak_labels = int(dist/100*len(self.coco_labels)
                               ) if self.epoch < 10 else 10**5
        n_of_coco_labels = len(self.coco_labels)

        print("N of weak labels: ", n_of_weak_labels)
        print("N of COCO labels: ", n_of_coco_labels)

        if self.group_iter != 0:
            weak_labels = self.weak_labels[self.group_iter*(n_of_weak_labels):self.group_iter*(n_of_weak_labels)+self.group_iter*(n_of_weak_labels)]
        else:
            weak_labels = self.weak_labels[:n_of_weak_labels]

        num_coco_samples = int(
            n_of_coco_labels / self.samples_per_gpu / self.num_replicas * self.samples_per_gpu)
        num_weak_samples = int(
            n_of_weak_labels / self.samples_per_gpu / self.num_replicas * self.samples_per_gpu)

        self.num_samples = num_coco_samples + num_weak_samples

        self.total_size = self.num_samples*self.num_replicas

        if len(weak_labels) != 0:
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

            assert coco_samples_per_batch + \
                weak_samples_per_batch == self.samples_per_gpu

            print("COCO samples x batch: ", coco_samples_per_batch)
            print("Weak samples x batch: ", weak_samples_per_batch)

            coco_count = 0
            weak_count = 0

            indices = []
            while len(indices) != int(self.total_size/self.samples_per_gpu)*self.samples_per_gpu:
                batch = [
                    *list(self.coco_labels[coco_count:coco_count +
                                      coco_samples_per_batch]),
                    *list(weak_labels[weak_count:weak_count+weak_samples_per_batch])
                ]

                if len(batch) != self.samples_per_gpu:
                    print("Batch", len(batch), len(indices))
                    indices.extend(self.coco_labels[coco_count:])
                    indices.extend(weak_labels[weak_count:])
                    break

                indices.extend(batch)

                coco_count += coco_samples_per_batch
                weak_count += weak_samples_per_batch
        else:
            indices = self.coco_labels

        print("Num replicas: ", self.num_replicas)
        # Subsamples
        if not self.num_replicas == 1:
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]

        print(len(indices), self.num_samples)
        import pdb; pdb.set_trace()
        return iter(indices)


if __name__ == '__main__':
    unittest.main()
