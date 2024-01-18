from flask import Flask, render_template, request, url_for
import tensorflow as tf
app = Flask(__name__)
import os 

PORT = os.environ['PORT'] or 3000;
model = tf.keras.models.load_model('./Heart_Prediction_model_1.h5')
@app.route('/')
def home():
    pred = (model.predict(
        [[ 80. ,   4. , 145. , 564. ,   0. , 160. ,   1.6,   0. ,   7. ,
          1. ,   0. ,   1. ,   0. ,   0. ,   1. ,   0. ,   0. ,   0. ,
          1. ]]
    ))
    return f"{pred[0]}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)