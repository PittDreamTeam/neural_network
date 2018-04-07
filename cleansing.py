
"""Utilities for transforming raw images into
    the parts we need for training and operation."""

import unittest
import numpy
from PIL import Image

def grid(image, dimensions):
    """Breaks up `image` into several pieces,
    each of which have the dimensions `dimensions`.
    Returns a list of these images."""
    lst = []
    image_width, image_height = image.size
    given_width, given_height = dimensions
    for i in range(0, image_width, given_width):
        for j in range(0, image_height, given_height):
            box = (i, j, i+given_width, j+given_height)
            temp = image.crop(box)
            lst.append(temp)
    return lst

class TestGrid(unittest.TestCase):
    """Tests the `grid` function."""
    def test_small(self):
        """Input is an all black image."""
        matrix = numpy.zeros([128, 128, 3], dtype='uint8')
        image = Image.fromarray(matrix)
        pieces = grid(image, (32, 32))
        self.assertEqual(16, len(pieces))
        first = pieces[0]
        self.assertEqual(32, first.size[0])
        self.assertEqual(32, first.size[1])
        self.assertEqual((0, 0, 0), first.getpixel((0, 0)))

if __name__ == '__main__':
    unittest.main()
