from flask import Flask,render_template,request
import pickle

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()
app = Flask(__name__)
@app.route("/")
def home():
    path = "CovidDataset/Val/Covid/covid-19-pneumonia-8.jpg"
    clf.image_prediction_and_visualization(path)
    return "hello world"+str(clf.model.predict(img)[0][0]*100)

if __name__ == "__main__":
    app.run(debug=True)
