import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
modelstandar = pickle.load(open("modelstandar.pkl", "rb"))
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    featuresstandar=modelstandar.transform(features)
    prediction = model.predict(featuresstandar)
    return render_template("index.html", prediction_text = "The class of wine is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)