import io
import json

import flask
import tensorflow as tf

global model
model = tf.keras.models.load_model('./model')

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
use_gpu = True


@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        print("POST_method")
        body = json.loads(flask.request.get_data())
        print(body)
        text_lst = body["text"]
        print(text_lst)

        # to do 
        # results = model.call(text_lst[0])
        # for r in results:
        #     data['predictions'].append(r)

        # Indicate that the request was a success.
        data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading model and Flask starting server ...")
    print("Please wait until server has fully started")

    app.run()
