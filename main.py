import argparse
import warnings

warnings.filterwarnings('ignore')

from plant_disease.config import Config
from plant_disease.data import create_gens, split_data
from plant_disease.evaluate import evaluate, get_predictions, print_classification_report
from plant_disease.model import build_model
from plant_disease.train import train
from plant_disease.utils import save_class_dict, save_model
from plant_disease.visualize import plot_training, show_images


def parse_args():
    parser = argparse.ArgumentParser(description='Plant Disease Classification')
    parser.add_argument('--data-dir', type=str, help='Path to dataset color directory')
    parser.add_argument('--save-path', type=str, help='Directory to save model outputs')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--freeze', action='store_true', help='Freeze base model, train head only')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.save_path:
        config.save_path = args.save_path
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.freeze:
        config.freeze_base = True

    train_df, valid_df, test_df = split_data(config.data_dir)
    train_gen, valid_gen, test_gen = create_gens(
        train_df, valid_df, test_df, config.batch_size, config.img_size
    )

    show_images(train_gen)

    model = build_model(len(train_gen.class_indices), config.img_shape, config.learning_rate, config.freeze_base)
    model.summary()

    history = train(model, train_gen, valid_gen, config)
    plot_training(history)

    train_score, valid_score, test_score = evaluate(model, train_gen, valid_gen, test_gen, test_df)

    y_pred = get_predictions(model, test_gen)
    print_classification_report(test_gen, y_pred)

    save_model(model, test_score, config.subject, config.save_path)
    save_class_dict(train_gen, config.subject, config.save_path)


if __name__ == '__main__':
    main()
