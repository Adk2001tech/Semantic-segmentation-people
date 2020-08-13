import os
import numpy as np

from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

from io import BytesIO
import base64

import load_net




app= Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/Prediction', methods = ['GET', 'POST'])
def pred():
    if request.method=='POST':
         file = request.files['file']
         org_img, feed_img= load_net.process_file(file)

         #uploaded image
         img_x0=BytesIO()
         plt.imshow(org_img)
         plt.savefig(img_x0,format='png')
         plt.close()
         img_x0.seek(0)
         plot_url0=base64.b64encode(img_x0.getvalue()).decode('utf8')

        #Model 1
         img = load_net.out_via_model_sm(feed_img)
         img_x=BytesIO()
         plt.imshow(img)
         plt.savefig(img_x,format='png')
         plt.close()
         img_x.seek(0)
         plot_url1=base64.b64encode(img_x.getvalue()).decode('utf8')

         #Model 2
         img2 = load_net.out_via_model_xl(feed_img)
         img_x2=BytesIO()
         plt.imshow(img2)
         plt.savefig(img_x2,format='png')
         plt.close()
         img_x2.seek(0)
         plot_url2=base64.b64encode(img_x2.getvalue()).decode('utf8')


    return render_template('pred.html', plot_url0=plot_url0,  plot_url1=plot_url1, plot_url2=plot_url2)


if __name__=='__main__':
    app.run()
