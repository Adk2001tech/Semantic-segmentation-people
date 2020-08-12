from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFile
from io import BytesIO

model_xl=load_model('models/unet_large.hdf5', compile=False)
model_sm=load_model('models/unet_small.h5', compile=False)


# reads from file object
# return array of original uploaded image and 1x128x128x3 processed image
def process_file(file):
    ImageFile.LOAD_TRUNCATED_IMAGES =False
    org_img=Image.open(BytesIO(file.read()))
    org_img.load()
    img=org_img.resize((128,128))

    img=image.img_to_array(img)/255.0
    org_img=image.img_to_array(org_img)/ 255.0

    return org_img, np.expand_dims(img,axis=0)


# return an ouptut (masked image)
def out_via_model_sm(img):
    mask=model_sm.predict(img)
    mask=np.float32((mask>0.7))       # Confidance level (0.7----70% confidant)
    img=img*mask                      #dim---- (1, 128, 128, 3)
    img= img[0]                       #dim-----(128,128,3)
    img[np.where((img == [0,0,0]).all(axis = 2))] = [1.0, 1.0, 1.0]
    return img


# return an ouptut (masked image)
def out_via_model_xl(img):
    mask=model_xl.predict(img)
    mask=np.float32((mask>0.7))      # Confidance level (0.7----70% confidant)
    img=img*mask                     #dim---- (1, 128, 128, 3)
    img= img[0]                       #dim-----(128,128,3)
    img[np.where((img == [0,0,0]).all(axis = 2))] = [1.0, 1.0, 1.0]
    return img
