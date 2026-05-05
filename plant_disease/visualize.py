import numpy as np
import matplotlib.pyplot as plt


def show_images(gen):
    classes = list(gen.class_indices.keys())
    images, labels = next(gen)
    sample = min(len(labels), 25)

    plt.figure(figsize=(20, 20))
    for i in range(sample):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i] / 255)
        plt.title(classes[np.argmax(labels[i])], color='blue', fontsize=12)
        plt.axis('off')
    plt.show()


def plot_training(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']

    index_loss = np.argmin(val_loss)
    index_acc = np.argmax(val_acc)
    epochs = range(1, len(tr_acc) + 1)

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_loss[index_loss], s=150, c='blue', label=f'best epoch= {index_loss + 1}')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, val_acc[index_acc], s=150, c='blue', label=f'best epoch= {index_acc + 1}')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
