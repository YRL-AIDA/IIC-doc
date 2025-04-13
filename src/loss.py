import torch
import sys
import numpy as np
def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()
  p_i_j = compute_joint(x_out, x_tf_out)
  assert (p_i_j.size() == (k, k))

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  p_i_j = torch.clamp(p_i_j, min=EPS)
  p_j = torch.clamp(p_j, min=EPS)
  p_i = torch.clamp(p_i, min=EPS)

  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  return loss


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j


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

        #print(losses)
    return np.mean(losses)
