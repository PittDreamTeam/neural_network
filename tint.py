"""Methods for adding tinting to images."""
from enum import Enum
from PIL import Image
import numpy

class Color(Enum):
    """Enum to specify the color of the tint.
    Example: `Color.RED` specifies a red tint."""
    RED = 0
    GREEN = 1
    BLUE = 2

def tint(image, top_left, bottom_right, color):
    """Gives the given region a red tint, by cutting the intensity of other channels in half.
    'top_left' and 'bottom_right' are tuples of pixel coords, (col, row) or (x, y) positions.
    'color' is a 'tint.Color', which is described in its docstring."""
    array = numpy.array(image)
    min_col, min_row = top_left
    max_col, max_row = bottom_right
    dividers = (
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 1)
    )
    div = dividers[color.value]
    array[min_row:max_row, min_col:max_col, 0] //= div[0]
    array[min_row:max_row, min_col:max_col, 1] //= div[1]
    array[min_row:max_row, min_col:max_col, 2] //= div[2]
    return Image.fromarray(array)

def main():
    """main"""
    img = Image.open('data/image3.jpg')
    tint(img, (200, 100), (400, 300), Color.GREEN).show()

if __name__ == '__main__':
    main()
