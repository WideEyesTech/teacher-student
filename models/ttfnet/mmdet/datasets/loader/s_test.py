import unittest
import numpy as np
import torch
import math


class Test(unittest.TestCase):
    '''
    '''

    def __init__(self, *args, **kwargs):
        super(Test, self).__init__(*args, **kwargs)
        self.epoch = 12
        self.flag = np.random.randint(2, size=215000)
        self.total_size = 1000
        self.samples_per_gpu = 16
        self.rank = 1
        self.num_replicas = 2
        self.num_samples = int((self.total_size / self.samples_per_gpu / self.num_replicas) * self.samples_per_gpu)

        assert self.total_size == self.num_samples * self.num_replicas


    def test_custom_sampler(self):

        # deterministically shuffle based on epoch
        g=torch.Generator()
        g.manual_seed(self.epoch)

        coco_labels=np.where(self.flag == 0)[0]
        # Shuffle based on epoch
        coco_labels=coco_labels[torch.randperm(len(coco_labels), generator=g)]
        weak_labels=np.where(self.flag == 1)[0]
        # Shuffle based on epoch
        weak_labels=weak_labels[torch.randperm(len(weak_labels), generator=g)]

        if self.epoch > 10:
            dist=(50, 50)
        else:
            dist=(100-(5*self.epoch), (5*self.epoch))

        coco_samples_per_batch=int((dist[0]*self.samples_per_gpu)/100)
        weak_samples_per_batch=int((dist[1]*self.samples_per_gpu)/100)
        extra=int(self.samples_per_gpu - \
            (coco_samples_per_batch+weak_samples_per_batch))

        # Create batches
        indices=[]
        for x in range(self.total_size//self.samples_per_gpu):

            if len(indices) == self.total_size:
                return indices

            idx_coco=(x*coco_samples_per_batch)+1
            if idx_coco >= len(coco_labels):
                coco_labels=np.concatenate(coco_labels, coco_labels)

            indices.extend(coco_labels[idx_coco:idx_coco+coco_samples_per_batch])

            idx_weak=(x*weak_samples_per_batch)+1
            if idx_coco >= len(coco_labels):
                weak_labels=np.concatenate(weak_labels, weak_labels)

            indices.extend(weak_labels[idx_weak:idx_weak+weak_samples_per_batch+extra])


        # subsample
        offset=self.num_samples * self.rank
        indices=indices[offset:offset + self.num_samples]
        
        # assert len(indices) == self.num_samples

        # Just to test output
        indices=list(map(lambda x: "w" if x in weak_labels else "o", indices))
        import pdb; pdb.set_trace()
        return indices



if __name__ == '__main__':
    unittest.main()
