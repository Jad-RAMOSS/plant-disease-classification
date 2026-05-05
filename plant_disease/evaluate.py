import numpy as np
from sklearn.metrics import classification_report


def evaluate(model, train_gen, valid_gen, test_gen, test_df):
    test_steps = len(test_df) // test_gen.batch_size

    train_score = model.evaluate(train_gen, steps=test_steps, verbose=1)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=1)
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=1)

    print(f'Train Loss: {train_score[0]:.4f}  |  Train Accuracy: {train_score[1] * 100:.2f}%')
    print(f'Valid Loss: {valid_score[0]:.4f}  |  Valid Accuracy: {valid_score[1] * 100:.2f}%')
    print(f'Test  Loss: {test_score[0]:.4f}  |  Test  Accuracy: {test_score[1] * 100:.2f}%')

    return train_score, valid_score, test_score


def get_predictions(model, test_gen) -> np.ndarray:
    preds = model.predict(test_gen)
    return np.argmax(preds, axis=1)


def print_classification_report(test_gen, y_pred: np.ndarray):
    classes = list(test_gen.class_indices.keys())
    print(classification_report(test_gen.classes, y_pred, target_names=classes))
