import pickle
from PIL import Image
import model

net = pickle.load( open( "net.pickle", "rb" ) )
im = Image.open("picture.jpg")
cars = model.find_cars(net, im)
print(cars)
