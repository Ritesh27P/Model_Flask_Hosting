from flask import Flask, render_template, request, url_for
import json
import tensorflow as tf
app = Flask(__name__)
import os 

PORT = 3000
# PORT = os.environ['PORT'] or 3000
model = tf.keras.models.load_model('./Heart_Prediction_model_1.h5')
@app.route('/')
def home():
    pred = (model.predict(
        [[ 80. ,   4. , 145. , 564. ,   0. , 160. ,   1.6,   0. ,   7. ,
          1. ,   0. ,   1. ,   0. ,   0. ,   1. ,   0. ,   0. ,   0. ,
          1. ]]
    ))
    return f"{pred[0]}"

@app.route('/heart', methods=['POST'])
def heart():
    data = request.get_json()
    inputs = request.json['inputs']
    inputs = [float(i) for i in inputs]
    hp_model = tf.keras.models.load_model('./Heart_Prediction_model_1.h5')
    # pred = hp_model.predict([float(data.inputs)])
    # print(json.loads(data))
    pred = hp_model.predict([inputs])
    value = pred[0][0]
    return {'value': float(value)}

@app.route('/disease', methods=['POST'])
def disease():
    data = request.json['symptoms']
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)