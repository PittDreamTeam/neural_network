"""How to train your neural network."""

import pickle
import unittest
import numpy
import cleansing
from PIL import Image, ImageDraw
from network import Network
from getimage3 import Camera

def lowres(pics):
    """Training images were captured at 640x480 res,
    but we want to work with 160x120 in operation.
    Takes a list of images and reduces both dimensions by a quarter."""
    res = []
    for pic in pics:
        width, height = pic.size
        res.append(pic.resize((width//4, height//4)))
    return res

def draw_grid(image):
    """Draws a grid on the given image.
    Credit https://randomgeekery.org/2017/11/24/drawing-grids-with-python-and-pillow/"""
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = image.width // 8

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width
    step_size = image.height // 8

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)
    return image

def read_labels(filename):
    """Reads the csv file containing the labels format,
    parses it to be accepted by `build_dataset`."""
    outer = []
    with open(filename) as csv:
        for line in csv:
            inner = []
            for elem in line.split(','):
                inner.append(int(elem.strip()))
            outer.append(inner)
    mat = numpy.array(outer).T
    res = []
    for line in mat.tolist():
        res += line
    return res

def construct_dataset(numbers):
    """Builds an entire dataset based on which images you are interested in.
    `numbers` is an iterable."""
    dataset = []
    for num in numbers:
        pic = Image.open('data/image{i}.jpg'.format(i=num))
        parts = lowres(cleansing.grid(pic, (80, 60)))
        dataset += cleansing.build_dataset(
            cleansing.compress(parts),
            read_labels('data/labels{i}.csv'.format(i=num))
        )
    return dataset

def draw_x(image, block):
    """Draws an 'X' on the specified block of the image."""
    max_x, max_y = 80, 60
    y, x = (block % 8), (block // 8)
    draw = ImageDraw.Draw(image)
    upward = (
        (x*max_x, y*max_y), # bottom-left
        ((x+1)*max_x, (y+1)*max_y) # top-right
    )
    draw.line(upward, fill=0xff0000)
    downward = (
        (x*max_x, (y+1)*max_y), # top-left
        ((x+1)*max_x, y*max_y) # bottom-right
    )
    draw.line(downward, fill=0xff0000)
    return image

def is_car(net, example):
    """Simply returns a bool indicating whether this
    `example` is a match on this `net`."""
    return abs(1 - net.feedforward(example)) < 0.4

def find_cars(net, image):
    """Apply the neural net to all pieces of `image`;
    return the indices of the pieces which are considered cars."""
    pics = cleansing.compress(
        lowres(cleansing.grid(image, (80, 60))))
    winners = []
    for i, pic in enumerate(pics):
        if is_car(net, pic):
            winners.append(i)
    return winners

def mark_cars(net, image):
    """Automatically breaks up the image,
    finds all the cars in it, and marks where they are."""
    winners = find_cars(net, image)
    draw_grid(image) # have to draw the grid after
    for i in winners: # recognition, else every cell
        draw_x(image, i) # will be considered "car".
    return image

def organize_grid(grid):
    """Turns a flat list of image segments into a list of lists of segments.
    Assumes that the grid is square,
    and that the pieces come in coluumn-major order.
    This output matrix is now in ROW-MAJOR order."""
    import math
    height = int(math.sqrt(len(grid)))
    res = [[] for _ in range(height)]
    for i, piece in enumerate(grid):
        res[i % height].append(piece)
    return res


def find_runs(lane, func, win_len=2):
    """Indentifies the positions in `lane` which
    satisfy `func` for at least `win_len` elements."""
    pos = []
    current_len = 0
    for i, elem in enumerate(lane):
        current_len = current_len + 1 if func(elem) else 0
        if current_len == win_len:
            pos.append(i - win_len + 1)
            current_len = 0
    return pos

class TestFindRuns(unittest.TestCase):
    """Tests the function `find_runs`."""
    def __init__(self, *args, **kwargs):
        """Not too interesting, just sets objective function."""
        super(TestFindRuns, self).__init__(*args, **kwargs)
        self.func = lambda x: x == 0
    def test_single_run(self):
        """Very simple test, one single run, default win length"""
        lane = [1, 1, 1, 0, 0, 1, 1]
        res = find_runs(lane, self.func)
        self.assertEqual(1, len(res))
        self.assertEqual(3, res[0])
    def test_two_separated_runs(self):
        """Now there are two runs, but they are separated by non-winners."""
        lane = [1, 0, 0, 1, 1, 0, 0, 1, 1]
        res = find_runs(lane, self.func)
        self.assertEqual(2, len(res))
        self.assertEqual(1, res[0])
        self.assertEqual(5, res[1])
    def test_two_back2back(self):
        """Now there are two runs, but they are back-to-back."""
        lane = [1, 1, 0, 0, 0, 0, 1, 1]
        res = find_runs(lane, self.func)
        self.assertEqual(2, len(res))
        self.assertEqual(2, res[0])
        self.assertEqual(4, res[1])
    def test_modified_win_len(self):
        """The same data as last time, except now the win length is 3."""
        lane = [1, 1, 0, 0, 0, 0, 1, 1]
        res = find_runs(lane, self.func, win_len=3)
        self.assertEqual(1, len(res))
        self.assertEqual(2, res[0])

def image2datagrid(image):
    """Input is a Python image.
    Output is a grid of numpy arrays which are the sections of our image."""
    return organize_grid(cleansing.compress(
        lowres(cleansing.grid(image, (80, 60)))
    ))

class Model:
    """A handy-dandy way to deal with our neural network and image processing."""
    def __init__(self, net, parking_lanes=(3, 6)):
        """Contruct a model.
        Param `net` is a `network.Network`, and must be opened by some fasion.
        `parking_lanes` is a pair of ints which specify
        the row of the two parking lanes in the image."""
        self.net = net
        self.top_lane = parking_lanes[0]
        self.low_lane = parking_lanes[1]
    def find_spaces(self, image, top_lane=False):
        """Takes an image, and returns the positions of the available spaces."""
        lane = self.top_lane if top_lane else self.low_lane
        return find_runs(
            image2datagrid(image)[lane],
            lambda x: not is_car(self.net, x)
        )

def main():
    """main"""
    cam = Camera()
    img = cam.take_photo()
    # net = pickle.load(open('net.pickle', 'rb'))
    # mark_cars(net, img).show()
    img.resize((160, 120)).save('forparth.jpg')

if __name__ == '__main__':
    unittest.main()
