import math

from .callbacks import MyCallback
from .config import Config


def train(model, train_gen, valid_gen, config: Config):
    batches = math.ceil(len(train_gen.labels) / config.batch_size)
    callbacks = [MyCallback(
        model=model,
        patience=config.patience,
        stop_patience=config.stop_patience,
        threshold=config.threshold,
        factor=config.factor,
        batches=batches,
        epochs=config.epochs,
        ask_epoch=config.ask_epoch,
    )]
    history = model.fit(
        x=train_gen,
        epochs=config.epochs,
        verbose=0,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=None,
        shuffle=False,
        workers=4,
        use_multiprocessing=False,
    )
    return history
