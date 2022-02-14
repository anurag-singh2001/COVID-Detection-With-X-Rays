import  numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from keras.models import Sequential,Model,load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import VGG16
import matplotlib.cm as cm
from IPython.display import Image, display
import pickle
from flask import Flask,render_template,request












# file=open('model.pkl','wb')
# pickle.dump(image_prediction_and_visualization,file)
# file.close()

# file=open('model.pkl','rb')
# clf=pickle.load(file)
# file.close()
app = Flask(__name__)
@app.route("/",methods=['GET','POST'])
def home():
    a=''
    b=''
    s=''
    model=load_model('model_vgg16.h5')
    class_type = {0:'Covid',  1 : 'Normal'}

    def get_img_array(img_path):
        path = img_path
        img = image.load_img(path, target_size=(224,224,3))
        img = image.img_to_array(img)/255
        img = np.expand_dims(img , axis= 0 )
        
        return img

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    def save_and_display_gradcam(img_path , heatmap, cam_path="./static/cam.jpg", alpha=0.4):
            # Load the original image
            img = keras.preprocessing.image.load_img(img_path)
            img = keras.preprocessing.image.img_to_array(img)

            
            # Rescale heatmap to a range 0-255
            heatmap = np.uint8(255 * heatmap)

            # Use jet colormap to colorize heatmap
            jet = cm.get_cmap("jet")

            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]

            # Create an image with RGB colorized heatmap
            jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
            jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * alpha + img
            superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

            # Save the superimposed image
            superimposed_img.save(cam_path)

            # Display Grad CAM
            display(Image(cam_path))
    def image_prediction_and_visualization(path,last_conv_layer_name = "block5_conv3", model = model):
            img=path
            img = image.load_img(path, target_size=(224,224,3))
            img = image.img_to_array(img)/255
            img = np.expand_dims(img , axis= 0 )
            # img_array = get_img_array(path)

            heatmap = make_gradcam_heatmap(path, model, last_conv_layer_name)

            #img = get_img_array(path)
            img=path
            img = image.load_img(path, target_size=(224,224,3))
            img = image.img_to_array(img)/255
            img = np.expand_dims(img , axis= 0 )

            res = class_type[np.argmax(model.predict(img))]
            print(f"The given X-Ray image is of type = {res}")
            print()
            s=res
            # print(f"The chances of image being Covid is : {model.predict(img)[0][0]*100} %")
            # print(f"The chances of image being Normal is : {model.predict(img)[0][1]*100} %")
            a=str((model.predict(img)[0][0]*100).item())
            print(a)
            b=str((model.predict(img)[0][1]*100).item())
            print(b)

            print()
            print("image with heatmap representing the covid spot")

            # function call
            save_and_display_gradcam(path, heatmap)

            print()
            print("the original input image")
            print()
            return a,b,s

            # a = plt.imread(path)
            # plt.imshow(a, cmap = "gray")
            # plt.title("Original image")
            # plt.show()
    if request.method == "POST":
        mydict = request.form
        print(request.form)
        path=mydict['image_input']
        print(type(path))
        a,b,s=image_prediction_and_visualization(path)
        # return f"The given X-Ray image is of type ={s}The chances of image being Covid is"+str(a)+"% and the chances of image being Normal is"+str(b)+"%"
    return render_template('index.html',a=a,b=b,s=s)
    print(s)
    print(a)
# @app.route('/')
# def display_image():
#     img_path='cam.jpg'
#     return render_template('index.html',img_path=img_path)   
    

        

if __name__ == "__main__":
    app.run(debug=True)
