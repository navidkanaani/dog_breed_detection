from crypt import methods
from distutils.log import debug
from flask import Flask, render_template, request
import estimator


# Load full trained model
loaded_full_model = estimator.load_model('./models/20201228-10091609150144-full-image-set-mobilenetv2-Adam.h5')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/estimator", methods=["POST"])
def predict():
    if request.method == "POST":
        img = request.files['my_image']
        img_path = "static/images/" + img.filename
        img.save(img_path)

        # Make prediction on custom images
        custom_data = estimator.create_data_batch(X=img_path)
        custom_pred = loaded_full_model.predict(custom_data)
        predicted_label = estimator.get_pred_label(custom_pred)
        print("Predicted label: %s" % predicted_label)

    return render_template("predicted.html", prediction = predicted_label, img_path = img_path)

if __name__=="__main__":
    app.run(debug=True)