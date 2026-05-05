import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


def find_model(outputs_dir: str = 'outputs') -> Tuple[str, str]:
    h5_files = [f for f in os.listdir(outputs_dir)
                if f.endswith('.h5') and 'weights' not in f]
    csv_files = [f for f in os.listdir(outputs_dir) if f.endswith('.csv')]
    if not h5_files:
        raise FileNotFoundError(f'No model .h5 found in {outputs_dir}/')
    if not csv_files:
        raise FileNotFoundError(f'No class dict CSV found in {outputs_dir}/')
    return os.path.join(outputs_dir, h5_files[0]), os.path.join(outputs_dir, csv_files[0])


def load_artifacts(model_path: str, class_dict_path: str):
    from tensorflow.keras.models import load_model as tf_load
    model = tf_load(model_path)
    df = pd.read_csv(class_dict_path)
    class_dict = dict(zip(df['class_index'].astype(int), df['class']))
    img_size = (int(df['width'].iloc[0]), int(df['height'].iloc[0]))  # (W, H) for cv2
    return model, class_dict, img_size


CONFIDENCE_THRESHOLD = 50.0


def predict_image(
    model,
    class_dict: Dict[int, str],
    img_size: Tuple[int, int],
    image_path: str,
) -> List[Dict]:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0).astype('float32')

    preds = model.predict(img, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:5]
    results = [
        {'class': class_dict[int(i)], 'confidence': float(preds[i]) * 100}
        for i in top_idx
    ]

    if results[0]['confidence'] < CONFIDENCE_THRESHOLD:
        return [{
            'class': 'Unidentified',
            'confidence': results[0]['confidence'],
            'low_confidence': True,
            'message': 'Confidence too low. Please try a clearer, higher-quality image.',
        }]

    return results
