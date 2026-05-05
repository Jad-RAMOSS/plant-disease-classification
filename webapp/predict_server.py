"""
Flask prediction server for the Plant Disease Image Classifier.
Loads the model once at startup; auto-discovers the .h5 file in outputs/.
Replace the .h5 file and restart to use a different model.
"""
import os
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Make the parent plant_disease package importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify  # noqa: E402
from plant_disease.predict import find_model, load_artifacts, predict_image  # noqa: E402

app = Flask(__name__)

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

print('Loading model…', flush=True)
_model_path, _class_dict_path = find_model(OUTPUTS_DIR)
_model, _class_dict, _img_size = load_artifacts(_model_path, _class_dict_path)
print(f'Ready — model: {os.path.basename(_model_path)}', flush=True)


@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    f = request.files['image']
    ext = os.path.splitext(f.filename)[1] if f.filename else '.jpg'

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = predict_image(_model, _class_dict, _img_size, tmp_path)
        return jsonify({'predictions': results})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model': os.path.basename(_model_path),
        'classes': len(_class_dict),
        'img_size': list(_img_size),
    })


@app.route('/classes', methods=['GET'])
def classes():
    return jsonify({
        'classes': [
            {'index': idx, 'name': name}
            for idx, name in sorted(_class_dict.items())
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False)
