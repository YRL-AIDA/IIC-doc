
from IPython.display import clear_output
import matplotlib.pyplot as plt
import json

def print_while_trainig(epochs_list, loss_history, loss_history_overclustering, save_to_jpg=False):
    """
    Выводит значения потерь и потерь от перекластеризации

    Параметры
    ----------
    epochs_list : список целых чисел
        Эпохи, для которых доступны значения потерь.
    loss_history: список чисел
        Значения потерь, размер списка соответствует размеру epochs_list.
    loss_history_overclustering : список чисел с плавающей запятой
        Значения потерь от перекластеризации, размер списка соответствует размеру epochs_list.
    """


    clear_output(True)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
    fig.set_figwidth(12)

    ax1.plot(epochs_list, loss_history, label="train_loss")
    ax1.legend()
    ax1.grid()

    ax2.plot(
        epochs_list, loss_history_overclustering, label="train_loss_overclustering"
    )
    ax2.legend()
    ax2.grid()
    if(save_to_jpg==False):
        plt.show()
    if(save_to_jpg==True):
        plt.savefig('../last_train/history.jpg')


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config