import flask
import os
import secrets
from flask_cors import CORS, cross_origin

from model import calculate_cosine,get_model

DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../data/clean"))

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.secret_key = "1234567890"

MODEL = None

sim_cache = {}

@app.route("/")
@cross_origin()
def base():
    return flask.send_from_directory("../client/dist", "index.html")

@app.route("/<path:path>")
def home(path):
    return flask.send_from_directory("../client/dist", path)

@app.route("/images/<path:path>")
def images(path):
    return flask.send_from_directory(DATA_DIR, path)

@app.route("/api/get-groups",methods=["GET"])
def get_available_groups():
    groups = os.listdir(DATA_DIR)
    return flask.jsonify({"groups":groups})

@app.route("/api/get-group-images",methods=["POST"])
def get_group_images():
    data = flask.request.json
    if "group" not in data:
        return flask.jsonify({"error": "No group provided"})

    gid = str(data["group"])
    ret_json = {x:[] for x in ["web","ai","final"]}
    for source in ["web","ai","final"]:
        for f in os.listdir(os.path.join(DATA_DIR,gid,source)):
            ret_json[source].append(os.path.join(gid,source,f))
    return flask.jsonify(ret_json)

@app.route("/api/get-similarity",methods=["POST"])
def get_sim():
    data = flask.request.json
    if "im1" not in data or "im2" not in data:
        return flask.jsonify({"error": "No filename provided"})
    k1,k2 = f"{data['im1']}--{data['im2']}",f"{data['im2']}--{data['im1']}"
    if k1 in sim_cache:
        return flask.jsonify({"similarity":sim_cache[k1]})
    if k2 in sim_cache:
        return flask.jsonify({"similarity":sim_cache[k2]})
    
    sim = calculate_cosine(os.path.join(DATA_DIR,data["im1"]),os.path.join(DATA_DIR,data["im2"]),MODEL)
    sim_cache[k1] = sim
    sim_cache[k2] = sim
    return flask.jsonify({"similarity":sim})

if __name__ == "__main__":
    MODEL = get_model()
    app.run("127.0.0.1",port=8080)