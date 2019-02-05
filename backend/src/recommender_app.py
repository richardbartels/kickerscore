from flask import Flask, jsonify, request
import numpy as np
import itertools
import json
from keras.models import load_model

app = Flask(__name__)

model = load_model('recommender_model.h5')
model._make_predict_function()
# print(model.summary())

with open('recommender_id2int.json', 'r') as fp:
    id2int = json.load(fp)

# inverse mapping
int2id = {v: k for k, v in id2int.items()}

@app.route('/', methods=['POST'])
def get_recommendation():
    # Check if the application is a json
    if request.headers['Content-Type'] != 'application/json':
        abort(400) # bad request

    try:
        d_ids = request.json
        # x_id = np.array([key for key in d_id])
        x_id = np.array(d_ids['slack_ids'])

    except KeyError:
        print("KeyError")
        abort(400)


    # convert slack ids to integers
    # x = np.array([id2int[i] for i in x_id])
    x = np.array([id2int.get(i, 0) for i in x_id]) # Return 0 when id not in the dictionary

    # Permute all ids, so  that the input is independent of the ordering
    x = np.array(list(itertools.permutations(x)))
    len_inputs = x.shape[1]

    # make the prediction
    y_pred = model.predict(x)[:, len_inputs - 1, :].mean(axis=0) # only return the last predcition in the sequen
    y_int = list(y_pred.argsort()[::-1] + 1) # convert prediction to integer  id

    # Remove input players and keep 5 suggestions
    for i in np.unique(x):
        if i in y_int:
            y_int.remove(i)
    y_int = y_int[:5] # keep first five items


    # convert integers back to ids
    y_id = [int2id[i] for i in y_int]
    y_id = {'slack_ids' : y_id}
    return json.dumps(y_id)

@app.route('/_ah/health')
def health():
    return "OK =)"



if __name__=='__main__':
    app.run(debug=False, host="0.0.0.0", port=8080, threaded=True)