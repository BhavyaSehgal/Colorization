#importing libraries
import numpy as np
import cv2

#paths to caffe model and image from github repo
prototxt = "colorization_deploy_v2.prototxt"
model = "colorization_release_v2.caffemodel"
kernel = "pts_in_hull.npy"
image_path = "gray images/face_2.jpg"

#Reads a network model stored in Caffe model in memory.... loads the cluster center points
net = cv2.dnn.readNetFromCaffe(prototxt,model)
#Load arrays or pickled objects from .npy, .npz or pickled files.
points = np.load(kernel)
 
#1,1 in argument is because treat each of the points as 1×1 convolutions and add them to the model.
points = points.transpose().reshape(2,313,1,1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

""" Like RGB, lab colour has 3 channels L, a, and b.
But here instead of pixel values, these have different significances i.e : 
L-channel: light intensity
a channel: green-red encoding
b channel: blue-red encoding """

gray_image = cv2.imread(image_path)
#normalize pixels intensities between 0 and 1 values
scaled = gray_image.astype("float32") / 255.0
#Converting from BGR to Lab color space
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

#resize the Lab image to 224x224 (the dimensions the colorization network accepts)
resized = cv2.resize(lab, (224, 224))
# split the L channel
L = cv2.split(resized)[0]
#perform mean centering by subtracting mean value...can change this value
L -= 50

#pass the L channel through the network to predict a and b values of image
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

#resize predicted image to original input image size
ab = cv2.resize(ab, (gray_image.shape[1], gray_image.shape[0]))
#get l value from  original input image
L = cv2.split(lab)[0]
#concatenate l value from input image with predicted a and b values 
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
#convert image to bgr from lab
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
#clip any values that fall outside the range [0, 1]
colorized = np.clip(colorized, 0, 1)
#change the image to 0-255 range and convert it from float32 to int
colorized = (255 * colorized).astype("uint8")

#show the black and white and colored image
cv2.imshow("Black and white image", gray_image)
cv2.imshow("colorized image", colorized)
img_path = image_path.split("/")
path = 'colorized_photos/'+img_path[1]
cv2.imwrite(path, colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
