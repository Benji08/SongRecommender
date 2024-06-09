import pickle
import random
import threading
from flask import Flask, jsonify, request, abort, make_response
import os
import shutil

app = Flask(__name__)

predictions = {"regression": {}, "nn": {}}
data_load_lock = threading.Lock()
ab_test_enabled = True
default_group = "regression"


def assign_group():
    return random.choice(["regression", "nn"])


def load_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        abort(404, description=f"File not found: {file_path}")


def load_predictions():
    file_map = {"regression": {}, "nn": {}}
    base_dir = "data"

    for model_type in file_map.keys():
        model_dir = os.path.join(base_dir, model_type)
        if os.path.exists(model_dir):
            for file_name in os.listdir(model_dir):
                if file_name.endswith('.pkl'):
                    genre = file_name.split('_')[2]  # Extract genre from file name
                    file_path = os.path.join(model_dir, file_name)
                    file_map[model_type][genre] = file_path

    with data_load_lock:
        for model_type, genres in file_map.items():
            for genre, path in genres.items():
                predictions[model_type][genre] = load_data(path)


def safe_replace_file(src, dest):
    temp_path = dest + '.tmp'
    shutil.copy2(src, temp_path)
    os.rename(temp_path, dest)
    threading.Thread(target=load_predictions).start()


@app.route('/')
def index():
    user_group = request.cookies.get('user_group')
    if ab_test_enabled:
        if not user_group:
            user_group = assign_group()
            resp = make_response(f"You have been assigned to the {user_group} group.")
            resp.set_cookie('user_group', user_group, max_age=60 * 60 * 24 * 365)  # Set cookie for 1 year
            return resp
    else:
        user_group = default_group

    return f"You are in the {user_group} group."


@app.route('/<genre>')
def get_data(genre):
    user_group = request.cookies.get('user_group')
    if ab_test_enabled:
        if not user_group:
            user_group = assign_group()
    else:
        user_group = default_group

    if user_group not in predictions or genre not in predictions[user_group]:
        abort(404, description=f"No data found for {genre} in {user_group} model.")

    music_data = predictions[user_group][genre]
    return jsonify(music_data)


@app.route('/update/<model_type>/<genre>', methods=['POST'])
def update_data(model_type, genre):
    if model_type not in ["regression", "nn"]:
        abort(404, description=f"Invalid model type: {model_type}")

    file_path = request.args.get('filename')
    if not file_path:
        abort(400, description="Filename query parameter is required.")

    dest_path = f"data/{model_type}/top_10_{genre}_{model_type}.pkl"
    safe_replace_file(file_path, dest_path)
    return jsonify({"message": f"File for {model_type}/{genre} updated successfully."})


@app.route('/toggle_ab_test', methods=['POST'])
def toggle_ab_test():
    global ab_test_enabled, default_group
    ab_test_enabled = request.json.get('ab_test_enabled', ab_test_enabled)
    default_group = request.json.get('default_group', default_group)
    return jsonify({"ab_test_enabled": ab_test_enabled, "default_group": default_group})


if __name__ == '__main__':
    load_predictions()
    app.run(host='0.0.0.0', port=5001)
