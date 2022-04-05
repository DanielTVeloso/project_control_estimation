from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from project_control_effort_estimation.modules import model
from paste.translogger import TransLogger
from waitress import serve
import os

app = Flask(__name__, template_folder=os.getcwd()+ r"\web\templates", static_url_path='', static_folder=os.getcwd()+r"\web\static")
app.config['JSON_SORT_KEYS'] = False # keep JSON in the order in which it is written

CORS(app)

@app.route('/docs', methods=['GET'])
def docs():
    return render_template('index.html')

@app.route('/new-task', methods=['POST'])
def new_task():
    pred_results, status_code = model.predict_from_model(request.json, load_path='storage/models')

    return jsonify(pred_results), status_code

def run_server():
    global app

    serve(TransLogger(app), host='0.0.0.0', port=8080)