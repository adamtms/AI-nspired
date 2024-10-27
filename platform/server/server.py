import flask
import os
import secrets
from flask_cors import CORS, cross_origin

from model import calculate_cosine,make_heatmap,get_model

DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../data/clean"))
CACHE_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../data/clean/cache"))

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
    groups = [x for x in os.listdir(DATA_DIR) if x != "cache"]
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
    if "image1" not in data or "image2" not in data:
        return flask.jsonify({"error": "No filename provided"})
    
    sim = calculate_cosine(os.path.join(DATA_DIR,data["image1"]),os.path.join(DATA_DIR,data["image2"]),MODEL)
    return flask.jsonify({"similarity":sim})


@app.route("/api/get-heatmap",methods=["POST"])
def get_heatmap():
    data = flask.request.json
    if "image1" not in data or "image2" not in data:
        return flask.jsonify({"error": "No filename provided"})
    
    save_folder = os.path.join(CACHE_DIR,"heatmaps")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    path1 = os.path.join(DATA_DIR,data["image1"])
    path2 = os.path.join(DATA_DIR,data["image2"])
    
    filename = f'{save_folder}/{path1.replace("/","_")}_{path2.replace("/","_")}_heatmap.jpg'
    
    if not os.path.exists(filename):
        make_heatmap(path1,path2,save_folder,MODEL)
    
    return flask.jsonify({"heatmap":f"cache/heatmaps/{path1.replace('/','_')}_{path2.replace('/','_')}_heatmap.jpg"})
    

if __name__ == "__main__":
    MODEL = get_model()
    app.run("127.0.0.1",port=8080)