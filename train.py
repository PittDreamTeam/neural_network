import model
from PIL import Image
import pickle

net = pickle.load( open( "net.pickle", "rb" ) )
data_set = model.construct_dataset(range(10))
net.SGD(data_set, 100, 8, 1)
pickle.dump( net, open( "net.pickle", "wb" ) )
