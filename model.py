"""How to train your neural network."""

import cleansing
from PIL import Image, ImageDraw
from network import Network

def lowres(pics):
    """Training images were captured at 640x480 res,
    but we want to work with 160x120 in operation.
    Takes a list of images and reduces both dimensions by a quarter."""
    res = []
    for pic in pics:
        width, height = pic.size()
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

def main():
    """main"""
    pic = Image.open('data/image0.jpg')
    parts = lowres(cleansing.grid(pic, (80, 60)))
    training_data = cleansing.build_dataset(
        cleansing.compress(parts),
        [0, 0, 0, 0, 0, 0, 0, 0] +
        [0, 0, 0, 0, 0, 0, 0, 0] +
        [0, 0, 0, 0, 1, 1, 1, 0] +
        [0, 0, 0, 0, 1, 0, 0, 0] +
        [1, 1, 0, 1, 1, 0, 0, 0] +
        [1, 0, 0, 1, 1, 0, 0, 0] +
        [0, 0, 0, 1, 0, 0, 0, 0] +
        [0, 0, 0, 0, 0, 0, 0, 0]
    )
    net = Network([20*15, 20*15, 20, 15, 1])
    net.SGD(training_data, 100, 8, 1, training_data)

if __name__ == '__main__':
    main()
