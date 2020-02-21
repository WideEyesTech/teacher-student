import unittest
from clustering import combine
import numpy as np

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
            [22.0, 23.0, 77.0, 76.0],
            [2.0, 3.0, 57.0, 86.0]
        ])

        bboxesB = np.array([
            [20.0, 23.0, 73.0, 74.0],
            [4.0, 5.0, 55.0, 88.0]
        ])

        expected_result = (bboxesA + bboxesB)/2
        result = combine(bboxesA, bboxesB)

        self.assertTrue(np.all(result == expected_result))

    def test_all_a_overlaps_but_b(self):
        '''
        Test with all bboxesA overlapping (with IoU greater than Threshold) bboxesB, but with some 
        bboxes in B not present in A
        '''

        # We create two lists of random almost equal bboxes
        bboxesA = np.array([
            [22.0, 23.0, 77.0, 76.0],
            [2.0, 3.0, 57.0, 86.0]
        ])

        bboxesB = np.array([
            [20.0, 23.0, 73.0, 74.0],
            [4.0, 5.0, 55.0, 88.0],
            [48.0, 52.0, 115.0, 112.0],
            [14.0, 15.0, 155.0, 188.0]
        ])

        expected_result = (bboxesA + bboxesB[:2])/2
        expected_result = np.vstack((expected_result, bboxesB[2:]))

        result = combine(bboxesA, bboxesB)

        self.assertTrue(np.all(result == expected_result))
    
    def test_all_B_overlaps_but_a(self):
        '''
        Test with all bboxesB overlapping (with IoU greater than Threshold) bboxesA, but with some 
        bboxes in A not present in B
        '''

        # We create two lists of random almost equal bboxes
        bboxesB = np.array([
            [22.0, 23.0, 77.0, 76.0],
            [2.0, 3.0, 57.0, 86.0]
        ])

        bboxesA = np.array([
            [20.0, 23.0, 73.0, 74.0],
            [4.0, 5.0, 55.0, 88.0],
            [48.0, 52.0, 115.0, 112.0],
            [14.0, 15.0, 155.0, 188.0]
        ])

        expected_result = (bboxesB + bboxesA[:2])/2
        expected_result = np.vstack((expected_result, bboxesA[2:]))

        result = combine(bboxesA, bboxesB)

        self.assertTrue(np.all(result == expected_result))
    
    def test_no_overlaps(self):
        '''
        Test with no overlaps between A and B
        '''

        # We create two lists of random almost equal bboxes
        bboxesA = np.array([
            [22.0, 23.0, 77.0, 76.0],
            [2.0, 3.0, 57.0, 86.0]
        ])

        bboxesB = np.array([
            [48.0, 52.0, 115.0, 112.0],
            [14.0, 15.0, 155.0, 188.0]
        ])

        expected_result = np.vstack((bboxesA, bboxesB))

        result = combine(bboxesA, bboxesB)

        self.assertTrue(np.all(result == expected_result))


if __name__ == '__main__':
    unittest.main()