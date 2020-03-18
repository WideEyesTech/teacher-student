import unittest
import numpy as np
import torch
import math


class Test(unittest.TestCase):
    '''
    '''

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.epoch = 2
        self.flag = np.random.randint(2, size=215000)
        self.total_size = 10**6
        self.samples_per_gpu = 16
        self.rank = 1
        self.num_replicas = 4
        self.num_samples = int(
            (self.total_size / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu)

        assert self.total_size == self.num_samples * self.num_replicas

    def test_custom_sampler(self):

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        coco_labels = np.where(self.flag == 0)[0]
        # Shuffle based on epoch
        coco_labels = np.array(coco_labels[torch.randperm(
            len(coco_labels), generator=g)])

        weak_labels = np.where(self.flag == 1)[0]
        # Shuffle based on epoch
        weak_labels = np.array(weak_labels[torch.randperm(
            len(weak_labels), generator=g)])

        if self.epoch > 10:
            dist = (50, 50)
        else:
            dist = (100-(5*self.epoch), (5*self.epoch))

        coco_samples_per_batch = int((dist[0]*self.samples_per_gpu)/100)
        weak_samples_per_batch = int((dist[1]*self.samples_per_gpu)/100)
        extra = int(self.samples_per_gpu -
                    (coco_samples_per_batch+weak_samples_per_batch))

        weak_samples_per_batch += extra

        assert coco_samples_per_batch + \
            weak_samples_per_batch == self.samples_per_gpu

        # Create batches
        indices = []
        count_coco = 0
        count_weak = 0
        while len(indices) != self.total_size:
            batch = []

            idx_coco = (count_coco*coco_samples_per_batch)

            if idx_coco+coco_samples_per_batch >= len(coco_labels):
                count_coco = 0
                idx_coco = (count_coco*coco_samples_per_batch)

            batch.extend(
                coco_labels[idx_coco:idx_coco+coco_samples_per_batch])

            idx_weak = (count_weak*weak_samples_per_batch)

            if idx_weak+weak_samples_per_batch >= len(weak_labels):
                count_weak = 0
                idx_weak = (count_weak*weak_samples_per_batch)

            batch.extend(
                weak_labels[idx_weak:idx_weak+weak_samples_per_batch])

            try:
                assert len(batch) == self.samples_per_gpu
            except AssertionError:
                import pdb; pdb.set_trace()

            indices.extend(batch)
            count_coco += 1
            count_weak += 1

        try:
            assert len(indices) == self.total_size
        except AssertionError:
            import pdb; pdb.set_trace()

        # subsample
        if not self.num_samples == self.total_size:
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]

        try:
            assert len(indices) == self.num_samples
        except AssertionError:
            import pdb; pdb.set_trace()

        return iter(indices)


if __name__ == '__main__':
    unittest.main()


