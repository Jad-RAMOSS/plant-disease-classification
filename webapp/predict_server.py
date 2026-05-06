"""
Flask prediction server for the Plant Disease Image Classifier.
Loads the model once at startup; auto-discovers the .h5 file in outputs/.
Replace the .h5 file and restart to use a different model.
"""
import os
import sys
import tempfile
import warnings
import requests as http_requests

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')

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


@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 204

    if not DEEPSEEK_API_KEY:
        return jsonify({'error': 'Chat service not configured — add DEEPSEEK_API_KEY to webapp/.env'}), 503

    data = request.get_json(silent=True) or {}
    disease = str(data.get('disease', 'Unknown plant disease'))[:200]
    history = data.get('history', [])

    clean_history = [
        {'role': m['role'], 'content': str(m['content'])[:2000]}
        for m in history
        if isinstance(m, dict)
        and m.get('role') in ('user', 'assistant')
        and 'content' in m
    ]

    system_prompt = (
        f'You are a concise plant disease expert. '
        f'The user\'s plant was diagnosed as "{disease}". '
        f'Answer questions about symptoms, causes, spread, treatment, and prevention. '
        f'Keep responses practical and under 150 words unless more detail is requested.'
    )

    messages = [{'role': 'system', 'content': system_prompt}] + clean_history

    try:
        resp = http_requests.post(
            'https://api.deepseek.com/chat/completions',
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json',
            },
            json={'model': 'deepseek-chat', 'messages': messages, 'max_tokens': 512},
            timeout=30,
        )
        resp.raise_for_status()
        reply = resp.json()['choices'][0]['message']['content']
        return jsonify({'reply': reply})
    except http_requests.exceptions.Timeout:
        return jsonify({'error': 'Chat service timed out. Please try again.'}), 504
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False)
