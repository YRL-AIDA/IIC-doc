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

os.environ["NO_ALBUMENTATIONS_UPDATE"]='1'

def parse_args():
    parser = argparse.ArgumentParser(description="Run clustering model training with configurable parameters.")
    parser.add_argument("-epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("-batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("-aug_number", type=int, default=10, help="Number of augmentations")
    parser.add_argument("-aug_batch_size", type=int, default=256, help="Batch size for augmentation")
    parser.add_argument("-overcluster_period", type=int, default=20, help="Period for overclustering")
    parser.add_argument("-overcluster_ratio", type=float, default=0.5, help="Ratio for overclustering")
    parser.add_argument("-dataset_path", type=str, help="Path to dataset.zip")
    parser.add_argument("-icdar", type=bool, help="choose icdar dataset", default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if(args.icdar == False):
        
        extract_path = "dataset"
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    if(args.icdar == True):
        labels_path = f'{dataset_path}\\labels\\train.txt'

    dataset_np = RAMAug(
        alb_transforms=alb_transforms,
        aug_number=args.aug_number,
        target_dir=args.dataset_path,
        aug_batch_size=args.aug_batch_size,
        aug_num_workers=1,
    )
    
    dataset_train, dataset_val = stratified_split(dataset_np, train_size=0.95)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=1
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

    final_features = 393216
    cluster_head = nn.Linear(final_features, 3)
    overcluster_head = nn.Linear(final_features, 15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using %s." % device)
    model = ResNetClusterisator()
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
        epochs=args.epochs,
        lamb=1.2,
        overcluster_period=args.overcluster_period,
        overcluster_ratio=args.overcluster_ratio,
    )

if __name__ == "__main__":
    main()
