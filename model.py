"""How to train your neural network."""

import pickle
import sys
import numpy
import cleansing
from PIL import Image, ImageDraw
from network import Network

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

def find_cars(net, image):
    """Apply the neural net to all pieces of `image`;
    return the indices of the pieces which are considered cars."""
    pics = cleansing.compress(
        lowres(cleansing.grid(image, (80, 60))))
    winners = []
    for i, pic in enumerate(pics):
        error = abs(net.feedforward(pic) - 1)
        if error < 0.4:
            winners.append(i)
    return winners

def mark_cars(net, image):
    """Automatically breaks up the image,
    finds all the cars in it, and marks where they are."""
    draw_grid(image)
    for i in find_cars(net, image):
        draw_x(image, i)
    return image

def main():
    """main"""
    pass

if __name__ == '__main__':
    main()
