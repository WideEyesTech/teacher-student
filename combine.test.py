import unittest
from clustering import combine
import numpy as np
import cytools


def create_sample_bbox(bbox=[]):
    # Generate score values
    score = np.zeros(81)  # Score + numebr of teachers inferences param
    score[-1] = 1  # At least one teacher have made inference
    category = 7  # Random hardcoded category
    category_value = np.random.random_sample()
    score[category] = category_value

    # Generate bbox values
    if len(bbox) == 0:
        bbox = np.random.randint(1, 100, size=2)
        bbox = np.concatenate((bbox, bbox*2), axis=None)

    return np.concatenate((bbox, score), axis=None)

# Create n_of_bboxes with overlap smaller relative to bboxes than thr
def create_bbox_no_overlaps(bboxes, n_of_bboxes, thr=.5):
    result_correct = []
    while not len(result_correct):
        bboxes_test = np.array([])

        # Generate bbox values
        for i in range(n_of_bboxes):
            bbox = np.random.randint(1, 100, size=2)
            bbox = np.concatenate((bbox, bbox*2), axis=None)
            bbox = bbox.astype(float)

            try:
                bboxes_test = np.vstack((bboxes_test, bbox))
            except:
                bboxes_test = np.array(bbox)

        ious = cytools.bbox_overlaps(bboxes_test[:, :4], bboxes[:, :4])
        # [
        #   [0.4, 0.0, 0.05, 0.1, 0.9],
        #   [0.9, 0.23, 0.0, 0.1, 0.1],
        #   [0.6, 0.1, 0.2, 0.1, 0.1],
        # ]

        max_ious = ious.max(1)  # [0.9, 0.9, 0.6]

        oks = max_ious > thr
        # [True, True, False]

        max_iousB = ious.max(0)  # [0.9, 0.23, 0.2, 0.1, 0.9]

        oksB = max_iousB > thr
        # [True, False, False, False, True]

        if np.all(oks == False) and np.all(oksB == False):
            result_correct = bboxes_test

    return result_correct


class Test(unittest.TestCase):
    '''
    Combine methods tests
    '''

    def test_all_overlaps(self):
        '''
        Test with all bboxesA overlapping (with IoU greater than Threshold) bboxesB
        '''

        # We create two lists of random almost equal bboxes
        bboxesA = np.array([
            create_sample_bbox(),
            create_sample_bbox()
        ])

        bboxesB = bboxesA

        def make_assert(bboxesA, bboxesB):
            expected_result = (bboxesA + bboxesB)/2
            expected_result[:, -1] = bboxesA[:, -1] + bboxesB[:, -1]
            result = combine(bboxesA, bboxesB)

            self.assertTrue(np.all(result == expected_result))

            return result

        result = make_assert(bboxesA, bboxesB)

        # Also test with aditional teacher with same inferences
        result2 = make_assert(result, result)


    def test_all_a_overlaps_but_b(self):
        '''
        Test with all bboxesA overlapping (with IoU greater than Threshold) bboxesB, but with some 
        bboxes in B not present in A
        '''

        # We create two lists of random almost equal bboxes
        bboxesA = np.array([
            create_sample_bbox(),
            create_sample_bbox()
        ])

        bboxesB = np.array([
            *bboxesA,
            *[create_sample_bbox(x) for x in create_bbox_no_overlaps(bboxesA, 4)]
        ])

        def make_assert(bboxesA, bboxesB):
            # In this test we assume that all bboxes A are in bboxes B,
            # and only B have different bboxes
            n_equal_bboxes = len(bboxesA)

            expected_result = (bboxesA + bboxesB[:n_equal_bboxes])/2
            # Bbboxes with multiple teachers prediction
            expected_result[:, -1] = (bboxesA[:, -1] +
                                      bboxesB[:n_equal_bboxes, -1])
            expected_result = np.vstack(
                (expected_result, bboxesB[n_equal_bboxes:]))
            # Bboxes only predicted by teacher B
            expected_result[n_equal_bboxes:, -1] = 1

            result = combine(bboxesA, bboxesB)
            
            self.assertTrue(np.all(result == expected_result))
          
            return result

        result = make_assert(bboxesA, bboxesB)

        bboxesC = np.array([*result, *[create_sample_bbox(x)
                                       for x in create_bbox_no_overlaps(result, 2)]])

        # Also test with aditional teacher with same inferences
        result2 = make_assert(result, bboxesC)

    def test_all_B_overlaps_but_a(self):
        '''
        Test with all bboxesB overlapping (with IoU greater than Threshold) bboxesA, but with some
        bboxes in A not present in B
        '''

        # We create two lists of random almost equal bboxes
        bboxesB = np.array([
            create_sample_bbox(),
            create_sample_bbox()
        ])

        bboxesA = np.array([
            *bboxesB,
            *[create_sample_bbox(x) for x in create_bbox_no_overlaps(bboxesB, 2)]
        ])

        def make_assert(bboxesA, bboxesB):
            # In this test we assume that all bboxes A are in bboxes B,
            # and only B have different bboxes
            n_equal_bboxes = len(bboxesB)

            expected_result = (bboxesB + bboxesA[:n_equal_bboxes])/2
            # Bbboxes with multiple teachers prediction
            expected_result[:, -1] = (bboxesB[:, -1] +
                                      bboxesA[:n_equal_bboxes, -1])
            expected_result = np.vstack(
                (expected_result, bboxesA[n_equal_bboxes:]))
            # Bboxes only predicted by teacher B
            expected_result[n_equal_bboxes:, -1] = 1

            result = combine(bboxesA, bboxesB)

            self.assertTrue(np.all(result == expected_result))

            return result

        result = make_assert(bboxesA, bboxesB)

        bboxesC = np.array([*result, *[create_sample_bbox(x)
                                       for x in create_bbox_no_overlaps(result, 3)]])

        # Also test with aditional teacher with same inferences
        result2 = make_assert(bboxesC, result)

    def test_no_overlaps(self):
        '''
        Test with no overlaps between A and B
        '''

        # We create two lists of random almost equal bboxes
        bboxesA = np.array([
            create_sample_bbox(),
            create_sample_bbox()
        ])

        bboxesB = np.array([create_sample_bbox(x) for x in create_bbox_no_overlaps(
            bboxesA, 5)])  # 5 is a random number

        expected_result = np.vstack((bboxesA, bboxesB))

        result = combine(bboxesA, bboxesB)

        self.assertTrue(np.all(result == expected_result))

        bboxesC = np.array([create_sample_bbox(x)
                            for x in create_bbox_no_overlaps(result, 3)])

        expected_result_add = np.vstack((result, bboxesC))

        result_add = combine(result, bboxesC)

        self.assertTrue(np.all(result_add == expected_result_add))


if __name__ == '__main__':
    unittest.main()
