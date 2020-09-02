import os
import numpy as np

from flask import Flask, render_template, url_for, request
from flask import send_file                                      ###################################################NEW
from werkzeug.utils import secure_filename


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from io import BytesIO
import base64
import cv2

import load_net


# img= np.ones(shape=(20,20,3))


app= Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/Prediction', methods = ['GET', 'POST'])
def pred():
    if request.method=='POST':
         file = request.files['file']
         org_img, feed_img= load_net.process_file(file)



        #Model 1


         arr = load_net.out_via_model_sm(feed_img)

         arr=arr*255.0
         img = Image.fromarray(arr.astype('uint8'))

         file_object= BytesIO()

         
         img.save(file_object, format='png')
        
         file_object.seek(0)
         return 'DONE'
         
         #return send_file(file_object, mimetype='image/PNG',attachment_filename='pic.png',  as_attachment=True)
           



if __name__=='__main__':
    app.run(debug=True)

