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
         #plt.close()
         file_object.seek(0)
         #plot_url1=base64.b64encode(img_x.getvalue()).decode('utf8')
         return send_file(file_object, mimetype='image/PNG',attachment_filename='pic.png',  as_attachment=True)


    #return render_template('pred.html', plot_url0=plot_url0,  plot_url1=plot_url1 )









if __name__=='__main__':
    app.run(debug=True)



#
# from flask import Flask, send_file, Response, jsonify
# import jsonpickle
# from PIL import Image
# import numpy as np
# import io
# import base64
#
# app = Flask(__name__)
#
# raw_data = [
#     [[255,255,255],[0,0,0],[255,255,255]],
#     [[0,0,1],[255,255,255],[0,0,0]],
#     [[255,255,255],[0,0,0],[255,255,255]],
# ]
#
# @app.route('/', methods = ['GET', 'POST'])
# def image():
#     # my numpy array
#     arr = np.array(raw_data)
#
#     # convert numpy array to PIL Image
#     img = Image.fromarray(arr.astype('uint8'))
#
#     # create file-object in memory
#     file_object = io.BytesIO()
#
#     # write PNG in file-object
#     img.save(file_object, 'PNG')
#
#     # move to beginning of file so `send_file()` it will read from start
#     file_object.seek(0)
#     return send_file(file_object, mimetype='image/PNG',attachment_filename='pic.png',  as_attachment=True)
#
#
#
# app.run()
