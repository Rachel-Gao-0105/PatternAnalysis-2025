# Classifier for ADNI brain data based on the ConvNeXt

**Table of Contents**
  - [Model and Problem Description](#model-and-problem-description)
  - [Model Architecture](#model-architecture)
  - [About the Dataset](#about-the-dataset)
  - [Usage](#usage)
  - [Results](#results)
  - [References](#references)

## Model and Problem Description



## Model Architecture

![ConvNeXt Architecture](images/convnext_arch.png)

## About the Dataset



## Usage 
### Files Description
1. modules.py containing the source code of the components of ConvNeXt model. 
2. dataset.py containing the data loader for loading and preprocessing ANDI dataset. 
3. train.py containing the source code for training, validating, testing and saving the model. 
4. predict.py showing example usage of the trained model. 
5. README.md to sufficiently document the project.
6. images/ containing figures and files used in README.
   
### Dependencies

| Dependencies | Version | 
| :-----: | :-----: | 
| Python       | 3.9 |  
| torchvision  | 0.22.1+cu118 | 
| torch        | 2.7.1+cu118 | 
| numpy        | 2.1.2 | 
| scikit-learn | 1.7.1 | 
| matplotlib   | 3.10.5 | 
| pillow       | 11.0.0 | 
| tqdm         | 4.67.1 | 

### Dataset Structure

AD_NC
├── train
│   ├── AD
│   │   ├── 218391_78.jpeg
│   │   ├── ...│
│   └── NC
│       ├── 808819_88.jpeg
│       ├── ...│
└── validation
    ├── AD
    │   ├── 388206_78.jpeg
    │   ├── ...│
    └── NC
        ├── 1182968_94.jpeg
        ├── ...

### Training

| Argument | Description | Type | Default |
| ----- | ----- | ----- | ----- |
| `--data_root`               | Root directory of the dataset, e.g., .../root/AD_NC | `str` | Required |
| `--img_size`                | Input resolution | `int` | `448` |
| `--num_classes`             | The number of output categories | `int` | `2` |
| `--drop_path_rate`          | Stochastic Depth (DropPath) rate | `float` | `0.1` |
| `--dropout_rate`            | Dropout probability in the classification head | `float` | `0.2` |
| `--weight_decay`            | Weight decay (L2 regularization) | `float` | `0.07` |
| `--label_smoothing`         | Label Smoothing factor for cross-entropy (set to `0` to disable) | `float` | `0.1` |
| `--batch_size`              | The number of samples per training iteration | `int` | `64` |
| `--epochs`                  | The number of training epochs | `int` | `50` |
| `--lr`                      | Initial learning rate | `float` | `1.5e-4` |
| `--out_dir`                 | Output directory |  `str` | `runs` |
| `--num_workers`             | The number of DataLoader workers | `int` | `0` |
| `--samples_per_class_train` | Number of samples to draw per class (for debugging; `None` to use all) | `int / None` |   `None` |
| `--seed`                    | Random seed | `int` | `42` |
| `--patience`                | Early stopping patience (set `0` to disable) |  `int` | `10` |
| `--target_acc`              | Early stopping target (set `0` to disable) | `float` | `0.8` |
| `--min_delta`               | Minimum delta for improvement | `float` | `1e-3` |
| `--mixup_alpha`             | MixUp coefficient (set `0` to disable) | `float` | `0.2` |

python train.py --data_root root/AD_NC --drop_path_rate 0.2 --dropout_rate 0.3 --lr 1e-4 --epochs 200 --batch_size 64 --num_workers 8 --patience 0 --mixup_alpha 0




### Predictions

| Argument | Description | Type | Default |
| ----- | ----- | ----- | ----- |
| `--ckpt`        | Path to the model weights, e.g., runs/best_acc_tuned.pth | `str` | Required |
| `--data_root`   | Directory of the dataset; used during batch evaluation, must contain subfolders `--split`  | `str` | `None` |
| `--split`       | The subdirectory under `data_root` for evaluation, e.g., `validation`/`test` | `str` | `validation` |
| `--image`       | Path for single-image inference | `str` | `None` |
| `--batch_size`  | Batch size for folder-based evaluation | `int` | `64` |
| `--num_workers` | The number of DataLoader workers | `int` | `0` |
| `--device`      | Device for Inference | `str` | `cuda` |
| `--save_dir`    | Output directory | `str` | `pred_out` |


python predict.py --ckpt runs5/best_acc_tuned.pth --data_root "root/AD_NC" --split validation --image "root/AD_NC/validation/NC/1182968_94.jpeg"

python predict.py --ckpt runs/best_acc_tuned.pth --data_root "root/AD_NC" --split validation


## Results
### Training
| Argument | Value |
| ----- | ----- |
| `--data_root`               | `root/AD_NC` |
| `--img_size`                | `448` |
| `--num_classes`             | `2` |
| `--drop_path_rate`          | `0.2` |
| `--dropout_rate`            | `0.3` |
| `--weight_decay`            | `0.07` |
| `--label_smoothing`         | `0.1` |
| `--batch_size`              | `64` |
| `--epochs`                  | `200` |
| `--lr`                      | `1e-4` |
| `--out_dir`                 | `runs` |
| `--num_workers`             | `8` |
| `--samples_per_class_train` | `None` |
| `--seed`                    | `42` |
| `--patience`                | `0` |
| `--target_acc`              | `0.8` |
| `--min_delta`               | `1e-3` |
| `--mixup_alpha`             | `0` |

python train.py --data_root root/AD_NC --img_size 448 --num_classes 2 --drop_path_rate 0.2 --dropout_rate 0.3 --weight_decay 0.07 --label_smoothing 0.1 --batch_size 64 --epochs 200 --lr 1e-4 --out_dir runs --num_workers 8 --samples_per_class_train None --seed 42 --patience 0 --target_acc 0.8 --min_delta 1e-3 --mixup_alpha 0

### Predictions

| Argument | Value |
| ----- | ----- |
| `--ckpt`        | `runs/best_acc_tuned.pth` |
| `--data_root`   | `root/AD_NC` |
| `--split`       | `validation` |
| `--image`       | `root/AD_NC/validation/NC/1182968_94.jpeg` |
| `--batch_size`  | `64` |
| `--num_workers` | `0` |
| `--device`      | `cuda` |
| `--save_dir`    | `pred_out` |


python predict.py --ckpt runs/best_acc_tuned.pth --data_root root/AD_NC --split validation --image root/AD_NC/validation/NC/1182968_94.jpeg --batch_size 64 --num_workers 0 --device cuda --save_dir pred_out


python predict.py --ckpt runs/best_acc_tuned.pth --data_root root/AD_NC --split validation --batch_size 64 --num_workers 0 --device cuda --save_dir pred_out




## References



