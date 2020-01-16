"""Unit testing of BasicDataLoader
"""
from unittest import TestCase, main, mock
# Paths to add
import sys
sys.path.insert(0, './')

from dataloader.basic import BasicDataLoader

class BasicDataLaoderTests(TestCase):
    """Main testing class for BasicDataLaoder
    """

    def test_len(self):
        """Test len() method
        """
        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = [
                'fake_img1.jpg', 'fake_img2.wrong extension', 'fake_img3.jpg']
            data_loader = BasicDataLoader('/fake/path')
            # Expect length to be 2
            # wrong extension file should be omitted
            self.assertEqual(data_loader.__len__(), 2)

    def test_getitem(self):
        """
        """

if __name__ == '__main__':
    main()
