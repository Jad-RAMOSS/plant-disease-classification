import time

import numpy as np
import tensorflow as tf
from tensorflow import keras


class MyCallback(keras.callbacks.Callback):
    def __init__(self, model, patience, stop_patience, threshold, factor, batches, epochs, ask_epoch):
        super().__init__()
        self.model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.batches = batches
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.ask_epoch_initial = ask_epoch

        self.count = 0
        self.stop_count = 0
        self.best_epoch = 1
        self.initial_lr = float(tf.keras.backend.get_value(model.optimizer.lr))
        self.highest_tracc = 0.0
        self.lowest_vloss = np.inf
        self.best_weights = self.model.get_weights()
        self.initial_weights = self.model.get_weights()

    def on_train_begin(self, logs=None):
        print('Do you want model asks you to halt the training [y/n] ?')
        ans = input('')
        self.ask_permission = 1 if ans in ['Y', 'y'] else 0
        self._print_header()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        elapsed = time.time() - self.start_time
        hours = elapsed // 3600
        minutes = (elapsed - hours * 3600) // 60
        seconds = elapsed - (hours * 3600 + minutes * 60)
        print(f'training elapsed time was {int(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)')
        self.model.set_weights(self.best_weights)

    def on_train_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy') * 100
        loss = logs.get('loss')
        print(
            f'{"":20s}processing batch {batch} of {str(self.batches):5s}'
            f'-   accuracy=  {acc:5.3f}   -   loss: {loss:8.5f}',
            '\r', end='',
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.ep_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.ep_start

        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        current_lr = lr
        acc = logs.get('accuracy')
        v_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        v_loss = logs.get('val_loss')

        if acc < self.threshold:
            monitor = 'accuracy'
            pimprov = 0.0 if epoch == 0 else (acc - self.highest_tracc) * 100 / self.highest_tracc
            if acc > self.highest_tracc:
                self.highest_tracc = acc
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                self.best_epoch = epoch + 1
            else:
                if self.count >= self.patience - 1:
                    lr *= self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                    self.count = 0
                    self.stop_count += 1
                    if v_loss < self.lowest_vloss:
                        self.lowest_vloss = v_loss
                else:
                    self.count += 1
        else:
            monitor = 'val_loss'
            pimprov = 0.0 if epoch == 0 else (self.lowest_vloss - v_loss) * 100 / self.lowest_vloss
            if v_loss < self.lowest_vloss:
                self.lowest_vloss = v_loss
                self.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                self.best_epoch = epoch + 1
            else:
                if self.count >= self.patience - 1:
                    lr *= self.factor
                    self.stop_count += 1
                    self.count = 0
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                else:
                    self.count += 1
                if acc > self.highest_tracc:
                    self.highest_tracc = acc

        print(
            f'{str(epoch + 1):^3s}/{str(self.epochs):4s} '
            f'{loss:^9.3f}{acc * 100:^9.3f}{v_loss:^9.5f}{v_acc * 100:^9.3f}'
            f'{current_lr:^9.5f}{lr:^9.5f}{monitor:^11s}{pimprov:^10.2f}{duration:^8.2f}'
        )

        if self.stop_count > self.stop_patience - 1:
            print(
                f' training has been halted at epoch {epoch + 1} after '
                f'{self.stop_patience} adjustments of learning rate with no improvement'
            )
            self.model.stop_training = True
        elif self.ask_epoch is not None and self.ask_permission != 0:
            if epoch + 1 >= self.ask_epoch:
                print('enter H to halt training or an integer for number of epochs to run then ask again')
                ans = input('')
                if ans in ['H', 'h']:
                    print(f'training has been halted at epoch {epoch + 1} due to user input')
                    self.model.stop_training = True
                else:
                    try:
                        self.ask_epoch += int(ans)
                        print(f' training will continue until epoch {self.ask_epoch}')
                        self._print_header()
                    except ValueError:
                        print('Invalid')

    def _print_header(self):
        print('{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^10s}{8:10s}{9:^8s}'.format(
            'Epoch', 'Loss', 'Accuracy', 'V_loss', 'V_acc', 'LR', 'Next LR', 'Monitor', '% Improv', 'Duration',
        ))
