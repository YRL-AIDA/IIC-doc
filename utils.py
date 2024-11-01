
from IPython.display import clear_output
import matplotlib.pyplot as plt

def print_while_trainig(epochs_list, loss_history, loss_history_overclustering):
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
    plt.show()