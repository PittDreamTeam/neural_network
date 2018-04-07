"""How to train your neural network."""

import cleansing
from PIL import Image
from network import Network

def main():
    """main"""
    pic = Image.open('data/image0.jpg')
    parts = cleansing.grid(pic, (80, 60))
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
    net = Network([80*60, 80*60, 1])
    input('done')
    net.SGD(training_data, 100, 8, 1, training_data)

if __name__ == '__main__':
    main()
