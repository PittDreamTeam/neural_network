import model
from PIL import Image
import pickle

net = pickle.load( open( "net.pickle", "rb" ) )
for i in range(10):
	im = Image.open("data/image{}.jpg".format(i))
	new_im = model.mark_cars(net, im)
	new_im.show()

