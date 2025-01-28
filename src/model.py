from torch import nn 
from torchvision import models 


import torch
from tqdm import tqdm
from loss import IID_loss, evaluate

from utils import print_while_trainig


class ResNetClusterisator(nn.Module):
    """Кластеризатор для IIC на основе основы ResNet18"""

    def __init__(self):
        super(ResNetClusterisator, self).__init__()
        # число выходов, посчитанное заранее
        final_features = 393216

        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-2]
        modules[0] = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        modules.append(nn.Flatten())
        self.backbone = nn.Sequential(*modules)

        
        self.cluster_head = nn.Linear(final_features, 16)
        self.overcluster_head = nn.Linear(final_features, 160)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, overclustering=False):
        """Прямой проход. Может работать как в режиме кластеризации, так и в режиме перекластеризации.

        Параметры
        ----------
        x : torch.tensor
            Входной батч.,
            где B — размер батча.
        overclustering : bool
            Если True, используется перекластеризации (overclustering head),
            в противном случае используется кластеризации (clustering head).
        """

        x = self.backbone(x)
        if overclustering:
            x = self.overcluster_head(x)
        else:
            x = self.cluster_head(x)

        return self.softmax(x)
    
def weight_init(model):
    """Initialises the model weights"""

    if isinstance(model, nn.Conv2d):
        nn.init.xavier_normal_(model.weight, gain=nn.init.calculate_gain("relu"))
        if model.bias is not None:
            nn.init.zeros_(model.bias)

    elif isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        if model.bias is not None:
            nn.init.zeros_(model.bias)

def choose_clustering_regime(epoch, overcluster_period, overcluster_ratio):
    """Выбор режима кластеризации на основе номера эпохи.

    Параметры
    ----------
    epoch : int
        Текущий номер эпохи.
    overcluster_period : int
        Общий период для кластеризации и перекластеризации.
    overcluster_ratio : float
        Доля времени, которое нужно потратить на перекластеризацию.
    """


    if (
        overcluster_period is not None
        and epoch % overcluster_period < overcluster_period * overcluster_ratio
    ):
        return True
    else:
        return False
    


def IIC_train(
    model,
    dataloader,
    optimizer,
    epochs=100,
    device=torch.device("cpu"),
    eval_every=5,
    lamb=1.0,
    overcluster_period=20,
    overcluster_ratio=0.5,
):
    """Цикл обучения IIC"""

    best_cluster_loss = float("inf")

    # Открываем файл для записи истории обучения
    with open("last_train.txt", "w") as log_file:
        for epoch in range(epochs):
            model.train()
            overclustering = choose_clustering_regime(
                epoch, overcluster_period, overcluster_ratio
            )

            # Прогресс-бар для текущей эпохи
            pbar = tqdm(total=len(dataloader), leave=True, desc=f"epoch#{epoch + 1}")
            epoch_loss = 0

            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                # Прямой проход
                inputs = batch["original"].to(device=device)
                inputs_tf = batch["aug"].to(device=device)

                outputs = model(inputs, overclustering)
                outputs_tf = model(inputs_tf, overclustering)
                loss = IID_loss(outputs, outputs_tf, lamb=lamb)

                # Обратный проход
                loss.backward()

                # Обновление весов
                optimizer.step()

                # Суммируем потери в эпохе
                epoch_loss += loss.item()

                # Обновляем прогресс-бар текущим значением loss
                pbar.set_postfix(loss=f"{loss.item():.6f}")
                pbar.update(1)

            # Средние потери по эпохе
            avg_loss = epoch_loss / len(dataloader)

            # Запись средней потери за эпоху в файл
            log_file.write(f"epoch#{epoch + 1} -- loss = {avg_loss:.6f}\n")
            log_file.flush()  # Немедленно записываем данные в файл

            # Сохранение модели при улучшении наилучшей потери
            if avg_loss < best_cluster_loss:
                best_cluster_loss = avg_loss
                torch.save(model.state_dict(), "best_loss_model")

            pbar.close()
