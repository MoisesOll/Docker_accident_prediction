import pickle
import inspect
from datetime import datetime
from pprint import pprint
import json
import numpy
import pandas
from flask_cors import CORS, cross_origin
from flask import Flask, Response, request
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class LoggerMiddleware(object):
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        print(
            f"{datetime.utcnow()} [V1] Request received from ",
            environ["REMOTE_ADDR"],
            " to resource: ",
            environ["RAW_URI"],
            flush=True,
        )
        return self.app(environ, start_response)


app = Flask(__name__)
app.wsgi_app = LoggerMiddleware(app.wsgi_app)
cors = CORS(app)

def process_request_arguments(request):
    sample = numpy.array(
        [
            request.args.get("C_VEHS_1"),
            request.args.get("C_VEHS_2"),
            request.args.get("C_VEHS_4"),
            request.args.get("C_VEHS_5"),
            request.args.get("C_VEHS_6"),
            request.args.get("C_CONF_1"),
            request.args.get("C_CONF_2"),
            request.args.get("C_CONF_3"),
            request.args.get("C_CONF_4"),
            request.args.get("C_CONF_5"),
            request.args.get("C_CONF_6"),
            request.args.get("C_CONF_7"),
            request.args.get("C_CONF_9"),
            request.args.get("C_CONF_10"),
            request.args.get("C_CONF_11"),
            request.args.get("C_CONF_12"),
            request.args.get("C_CONF_13"),
            request.args.get("C_CONF_14"),
            request.args.get("C_CONF_15"),
            request.args.get("C_CONF_17"),
            request.args.get("C_CONF_18"),
            request.args.get("C_RCFG_1"),
            request.args.get("C_RCFG_2"),
            request.args.get("C_RCFG_3"),
            request.args.get("C_RCFG_4"),
            request.args.get("C_RCFG_6"),
            request.args.get("C_RCFG_7"),
            request.args.get("C_RCFG_8"),
            request.args.get("C_RCFG_9"),
            request.args.get("C_RCFG_10"),
            request.args.get("C_RCFG_11"),
            request.args.get("C_RCFG_12"),
            request.args.get("C_RCFG_13"),
            request.args.get("C_RCFG_14"),
            request.args.get("C_RCFG_15"),
            request.args.get("C_RCFG_16"),
            request.args.get("C_RCFG_17"),
            request.args.get("C_RCFG_18"),
            request.args.get("C_WTHR_1"),
            request.args.get("C_WTHR_2"),
            request.args.get("C_WTHR_3"),
            request.args.get("C_WTHR_4"),
            request.args.get("C_WTHR_5"),
            request.args.get("C_RSUR_1"),
            request.args.get("C_RSUR_2"),
            request.args.get("C_RSUR_3"),
            request.args.get("C_RSUR_4"),
            request.args.get("C_RALN_1"),
            request.args.get("C_RALN_2"),
            request.args.get("C_RALN_3"),
            request.args.get("C_RALN_4"),
            request.args.get("C_RALN_5"),
            request.args.get("C_TRAF_2"),
            request.args.get("C_TRAF_3"),
            request.args.get("C_TRAF_4"),
            request.args.get("C_TRAF_5"),
            request.args.get("C_TRAF_6"),
            request.args.get("C_TRAF_7"),
            request.args.get("C_TRAF_8"),
        ]
    ).reshape(1, -1)
    return sample


@app.route("/accident/test", methods=["GET"])
def test():
    response = Response("This response is a test :)", status=200)
    return response


@app.route("/accident/lightgbm", methods=["GET"])
def handler_request_lightgbm():
    """
    Handles every request to /accident/lightgbm
    Needs the 59 parameters when fitted the model as input
    Returns the prediction as 0 or 1
    """
    # load the model as binary object
    lightg_gbm = pickle.load(open("./models/LightGBM-complete.pkl", "rb"))
    
    sample = process_request_arguments(request)
    print(f"{inspect.stack()[0][3]} with sample {sample}")
    prediction_value : int = int(lightg_gbm.predict(sample)[0])
    response_body = {
        'prediction': prediction_value
    }
    response = Response(json.dumps(response_body), status=200)
    return response


# Always put headers
@app.after_request
def apply_caching(response):
    response.headers["Content-Type"] = "application/json"
    response.headers.set('Access-Control-Allow-Methods', 'GET')
    return response

