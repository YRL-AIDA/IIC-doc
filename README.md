# IIC-doc

Required Python 3.12.7

run train on ICDAR dataset

```
python train.py -epochs 50 -batch_size 256 -aug_number 10 -aug_batch_size 256 -overcluster_period 10 -overcluster_ratio 0.5 -icdar 1 -dataset_path "<path to unziped dataset>\rvl-cdip"
```

full dataset source -- https://adamharley.com/rvl-cdip/
