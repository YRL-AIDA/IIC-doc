# IIC-doc

Required Python 3.12.7

run train

```
cd src
python train.py --config cfg.json
```

config example:

```
{
  "epochs": 100,
  "batch_size": 6,
  "aug_number": 2,
  "aug_batch_size": 16,
  "overcluster_period": 20,
  "overcluster_ratio": 0.5,
  "labels_path": "D:\\rvl-cdip.tar\\rvl-cdip\\labels\\train_3class_500.txt",
  "images_path": "D:\\rvl-cdip.tar\\rvl-cdip\\images\\",
  "aug_num_workers": 0,
  "class_num": 3
}
```
