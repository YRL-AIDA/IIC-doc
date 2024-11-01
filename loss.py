import torch
import sys
import numpy as np
def compute_joint(x_out, x_tf_out):
    """Оценивает совместное распределение вероятностей.

    Параметры
    ----------
    x_out : torch.tensor
        Форма (B, C), где B — размер батча, C — количество классов.
        Вероятности для исходного батча.
    x_out_tf : torch.tensor
        Та же форма, что и x_out.
        Вероятности для трансформированного батча.

    Возвращает
    -------
    p_i_j : torch.tensor
        Форма (C, C), где C — количество классов (такое же, как в x_out).
        Совместные вероятности.
    """

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k

    p_i_j = p_i_j.mean(dim=0)

    p_i_j = (p_i_j + p_i_j.t()) / 2.0

    return p_i_j


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Вычисляет потерю взаимной информации. Общий минус добавлен, поэтому потеря должна быть минимизирована.

    Параметры
    ----------
    x_out : torch.tensor
        Форма (B, C), где B — размер батча, C — количество классов.
        Вероятности для исходного батча.
    x_out_tf : torch.tensor
        Та же форма, что и x_out.
        Вероятности для трансформированного батча.
    lambd : float
        Параметр, модифицирующий потерю.
        Больший lambd обычно побуждает модель к размещению
        выборок в различные кластеры равного размера.
        Меньший lambd побуждает модель к размещению похожих изображений в один кластер.
    EPS : float
        Параметр для регуляризации малых вероятностей.

    Возвращает
    -------
    loss : torch.tensor
        shape (1,). Потеря взаимной информации.
    """

    _, num_classes = x_out.size()

    p_i_j = compute_joint(x_out, x_tf_out)
    assert p_i_j.size() == (num_classes, num_classes)

    mask = ((p_i_j > EPS).data).type(torch.float32)
    p_i_j = p_i_j * mask + EPS * (1 - mask)

    p_i = p_i_j.sum(dim=1).view(num_classes, 1).expand(num_classes, num_classes)
    p_j = p_i_j.sum(dim=0).view(1, num_classes).expand(num_classes, num_classes)

    loss = -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    loss = torch.sum(loss)

    return loss


def evaluate(
    model, dataloader, overclustering=False, lamb=1.0, device=torch.device("cpu")
):
    """Вычисляет среднюю потерю модели. Среднее значение берется по всем батчам."""

    losses = []
    model.eval()

    for i, batch in enumerate(dataloader):

        # forward run
        inputs = batch["original"]
        inputs_tf = batch["aug"]
        with torch.no_grad():

            inputs = inputs.to(device=device)
            inputs_tf = inputs_tf.to(device=device)

            outputs = model(inputs, overclustering)
            outputs_tf = model(inputs_tf, overclustering)

        loss = IID_loss(outputs, outputs_tf, lamb=lamb)

        losses.append(loss.data.cpu().numpy())

    return np.mean(losses)
