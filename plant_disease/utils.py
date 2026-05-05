import os

import pandas as pd


def save_model(model, test_score, subject: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    model_name = model.input_names[0][:-6]
    acc = test_score[1] * 100

    model_path = os.path.join(save_path, f'{model_name}-{subject}-{acc:.2f}.h5')
    model.save(model_path)
    print(f'model saved as {model_path}')

    weights_path = os.path.join(save_path, f'{model_name}-{subject}-weights.h5')
    model.save_weights(weights_path)
    print(f'weights saved as {weights_path}')


def save_class_dict(train_gen, subject: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    class_dict = train_gen.class_indices
    h, w = train_gen.image_shape[:2]
    n = len(class_dict)

    class_df = pd.concat([
        pd.Series(list(class_dict.values()), name='class_index'),
        pd.Series(list(class_dict.keys()), name='class'),
        pd.Series([h] * n, name='height'),
        pd.Series([w] * n, name='width'),
    ], axis=1)

    csv_path = os.path.join(save_path, f'{subject}-class_dict.csv')
    class_df.to_csv(csv_path, index=False)
    print(f'class dict saved as {csv_path}')
