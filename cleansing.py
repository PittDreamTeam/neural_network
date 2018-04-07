
"""Utilities for transforming raw images into
    the parts we need for training and operation."""

import unittest
import numpy
from PIL import Image

def generate_black_image():
    """Returns a black, RGB, 128x128 image."""
    matrix = numpy.zeros([128, 128, 3], dtype='uint8')
    return Image.fromarray(matrix)

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
        image = generate_black_image()
        pieces = grid(image, (32, 32))
        self.assertEqual(16, len(pieces))
        first = pieces[0]
        self.assertEqual(32, first.size[0])
        self.assertEqual(32, first.size[1])
        self.assertEqual((0, 0, 0), first.getpixel((0, 0)))

def matrix_from(value):
    """Transforms various inputs to `numpy.ndarray` with shape `(n, k)`.
    Becuase we usually want colomn vectors,
    lists and numpy vector will become such."""
    tov = type(value)
    if tov == int or tov == float:
        return numpy.array([[value]])
    elif tov == list:
        return numpy.array([value]).T
    elif tov == numpy.ndarray and len(value.shape) == 1:
        return numpy.array([value.tolist()]).T
    raise Exception("Invalid input type {}".format(tov))

class TestMatrixFrom(unittest.TestCase):
    """Yknow"""
    def test_number(self):
        """yeh"""
        res = matrix_from(4)
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual((1, 1), res.shape)
        self.assertEqual(4, res[0, 0])
    def test_list(self):
        """Lists should be column vectors."""
        res = matrix_from([1, 2, 3, 4])
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual((4, 1), res.shape)
        self.assertEqual(2, res[1, 0])
    def test_vector(self):
        """Vectors should become column vectors."""
        res = matrix_from(numpy.array([1, 2, 3, 4]))
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual((4, 1), res.shape)
        self.assertEqual(3, res[2, 0])

def build_dataset(inputs, outputs):
    """Matches elements from `inputs` to those in `outputs` into a list of tuples.
    Returns a list where each element is a tuple of the form:
        (inputs[i], outputs[i])
    such that our neural net should yeild `outputs[i]` when presented with `inputs[i]`.
    Note that `outputs` will be placed into numpy arrays, but the `inputs` type
    is the responsibility of the user."""
    return list(map(
        lambda tup: (tup[0], matrix_from(tup[1])),
        zip(inputs, outputs)
    ))

class TestBuildDataset(unittest.TestCase):
    """Tests the above."""
    def test_small(self):
        """Very simple problem."""
        inputs = [1, 2, 3, 4]
        outputs = [0, 0, 1, 0]
        uut = build_dataset(inputs, outputs)
        self.assertIsInstance(uut[0][1], numpy.ndarray)
        self.assertEqual((1, 1), uut[0][1].shape)
        self.assertEqual(0, uut[0][1][0, 0])

def compress(images):
    """Turns a list of images to a list of flat numpy arrays."""
    vectors = []
    for img in images:
        vectors.append(
            numpy.asarray(img.convert('L'), dtype='uint8')
            .flatten()
            / 255
        )
    return vectors

class TestCompress(unittest.TestCase):
    """Tests compress."""
    def test_simple(self):
        """The thing."""
        res = compress([generate_black_image()])[0]
        self.assertIsInstance(res, numpy.ndarray)
        self.assertEqual((128*128,), res.shape)
        self.assertEqual(0, res[234])

if __name__ == '__main__':
    unittest.main()
