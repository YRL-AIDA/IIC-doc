import zipfile
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch
from torch import nn

from data_processing import RAMAug, alb_transforms, stratified_split
from model import ResNetClusterisator, weight_init
from loss import IID_loss
from model import IIC_train

import os
import json

from utils import load_config



os.environ["NO_ALBUMENTATIONS_UPDATE"]='1'

def parse_args():
    parser = argparse.ArgumentParser(description="Run clustering model training with configurable parameters.")
    parser.add_argument('--config', type=str, required=True, help='path to configuration file')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    EPOCHS = cfg['epochs']
    BATCH_SIZE = cfg['batch_size']
    AUG_NUMBER = cfg['aug_number']
    AUG_BATCH_SIZE = cfg['aug_batch_size']
    OVERCLUSTER_PERIOD = cfg['overcluster_period']
    OVERCLUSTER_RATIO = cfg['overcluster_ratio']
    LABELS_PATH = cfg['labels_path']
    IMAGES_PATH = cfg['images_path']
    AUG_NUM_WORKERS = cfg['aug_num_workers']
    CLASS_NUM = cfg['class_num']


    if not os.path.exists('../last_train'):
        os.makedirs("../last_train")



    dataset_np = RAMAug(
        alb_transforms=alb_transforms,
        aug_number=AUG_NUMBER,
        labels_path=LABELS_PATH,
        images_path=IMAGES_PATH,
        aug_batch_size=AUG_BATCH_SIZE,
        aug_num_workers=AUG_NUM_WORKERS,
    )
    
    dataset_train, dataset_val = stratified_split(dataset_np, train_size=0.95)

    dataloader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )

    resnet = models.resnet18(pretrained=False)
    modules_to_keep = list(resnet.children())[:-2]
    modules_to_keep[0] = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    modules_to_keep.append(nn.Flatten())
    backbone = nn.Sequential(*modules_to_keep)

    batch = next(iter(dataloader_train))["original"]
    batch = batch[0:6]

    print("Batch shape:", batch.shape)
    print("Output shape:", backbone(batch).shape)

    final_features = backbone(batch).shape[-1]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using %s." % device)
    model = ResNetClusterisator(class_num=cfg['class_num'], final_features=final_features)
    model.to(device)
    model.apply(weight_init)
    print("The model is transferred to %s." % device)
    print("The weights are initialized.")

    batch = next(iter(dataloader_train))["original"]
    batch = batch.to(device)
    print("Model output shape in clustering mode:", model(batch).shape)
    print("Model output shape in overclustering mode:", model(batch).shape)

    batch = next(iter(dataloader_train))
    print("Batch entities:", batch.keys(), end="\n\n")
    print("Original images batch shape:     {0}".format(batch["original"].shape))
    print("Transformed images batch shape:  {0}".format(batch["aug"].shape))
    print("Labels batch shape:              {0}".format(batch["label"].shape))

    inputs = batch["original"].to(device=device)
    inputs_tf = batch["aug"].to(device=device)

    overclustering = False
    lamb = 1.0
    outputs = model(inputs, overclustering)
    outputs_tf = model(inputs_tf, overclustering)
    loss = IID_loss(outputs, outputs_tf, lamb=lamb)
    #print(loss.data.cpu().numpy())
    #loss.backward()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=4e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )
    IIC_train(
        model,
        dataloader_train,
        optimizer,
        device=device,
        epochs=EPOCHS,
        lamb=1.2,
        overcluster_period=OVERCLUSTER_PERIOD,
        overcluster_ratio=OVERCLUSTER_RATIO,
    )

if __name__ == "__main__":
    main()
