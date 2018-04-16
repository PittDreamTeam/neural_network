from PIL import Image
import numpy

def tintRed(im, minRow, maxRow, minCol, maxCol):
	a = numpy.array(im)
	#a[:,:,0] = numpy.clip(a[:,:,0]*2,0,256)
	
	#make red
	a[minRow:maxRow,minCol:maxCol,1] *=0
	a[minRow:maxRow,minCol:maxCol,2] *=0

	return Image.fromarray(a)

def tintGreen(im, minRow, maxRow, minCol, maxCol):
	a = numpy.array(im)
	#a[:,:,0] = numpy.clip(a[:,:,0]*2,0,256)
	
	#make green
	a[minRow:maxRow,minCol:maxCol,0] *=0
	a[minRow:maxRow,minCol:maxCol,2] *=0

	return Image.fromarray(a)

def tintBlue(im, minRow, maxRow, minCol, maxCol):
	a = numpy.array(im)
	#a[:,:,0] = numpy.clip(a[:,:,0]*2,0,256)
	
	#make red
	a[minRow:maxRow,minCol:maxCol,0] *=0
	a[minRow:maxRow,minCol:maxCol,1] *=0

	return Image.fromarray(a)

im = Image.open("picture.jpg")
im = tintGreen(im,100,200,100,200)
im.show()
