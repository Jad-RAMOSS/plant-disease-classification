import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def define_paths(data_dir: str) -> Tuple[list, list]:
    filepaths, labels = [], []
    for fold in os.listdir(data_dir):
        foldpath = os.path.join(data_dir, fold)
        if not os.path.isdir(foldpath):
            continue
        for file in os.listdir(foldpath):
            filepaths.append(os.path.join(foldpath, file))
            labels.append(fold)
    return filepaths, labels


def define_df(files: list, classes: list) -> pd.DataFrame:
    return pd.concat([
        pd.Series(files, name='filepaths'),
        pd.Series(classes, name='labels'),
    ], axis=1)


def split_data(data_dir: str):
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    train_df, dummy_df = train_test_split(
        df, train_size=0.8, shuffle=True, random_state=123, stratify=df['labels']
    )
    valid_df, test_df = train_test_split(
        dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=dummy_df['labels']
    )
    return train_df, valid_df, test_df


def create_gens(train_df, valid_df, test_df, batch_size: int, img_size: Tuple[int, int] = (224, 224)):
    ts_length = len(test_df)
    test_batch_size = max(
        sorted([ts_length // n for n in range(1, ts_length + 1)
                if ts_length % n == 0 and ts_length / n <= 80])
    )

    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    train_gen = tr_gen.flow_from_dataframe(
        train_df, x_col='filepaths', y_col='labels',
        target_size=img_size, class_mode='categorical',
        color_mode='rgb', shuffle=True, batch_size=batch_size,
    )
    valid_gen = ts_gen.flow_from_dataframe(
        valid_df, x_col='filepaths', y_col='labels',
        target_size=img_size, class_mode='categorical',
        color_mode='rgb', shuffle=True, batch_size=batch_size,
    )
    test_gen = ts_gen.flow_from_dataframe(
        test_df, x_col='filepaths', y_col='labels',
        target_size=img_size, class_mode='categorical',
        color_mode='rgb', shuffle=False, batch_size=test_batch_size,
    )
    return train_gen, valid_gen, test_gen
