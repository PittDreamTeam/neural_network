"""Methods for adding tinting to images."""
from PIL import Image
import numpy

class TintColor:
    """Quickly wrap up the color which you would like to tint an image.
    Set to True the color you're interested in; the others will be left False."""
    def __init__(self, red=False, green=False, blue=False):
        self.red = red
        self.green = green
        self.blue = blue

def tint(image, top_left, bottom_right, color):
    """Gives the given region a red tint, by cutting the intensity of other channels in half.
    'top_left' and 'bottom_right' are tuples of pixel coords, (col, row) or (x, y) positions.
    'color' is a 'TintColor', which is described in its docstring."""
    array = numpy.array(image)
    min_col, min_row = top_left
    max_col, max_row = bottom_right
    array[min_row:max_row, min_col:max_col, 0] //= (1 if color.red else 2)
    array[min_row:max_row, min_col:max_col, 1] //= (1 if color.green else 2)
    array[min_row:max_row, min_col:max_col, 2] //= (1 if color.blue else 2)
    return Image.fromarray(array)

def main():
    """main"""
    img = Image.open('data/image3.jpg')
    tint(img, (300, 100), (500, 300), TintColor(red=True)).show()

if __name__ == '__main__':
    main()
