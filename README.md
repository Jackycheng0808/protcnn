# Pytorch gene sequence classification by protCNN

This Repo is using various of CNN models to for Pfam Protein Sequence Annotation.

Model Inspired by ["Using Deep Learning to Annotate the Protein Universe"](https://www.biorxiv.org/content/10.1101/626507v3.full.pdf)

Above 97% test accuracy trained by resNet with Adam optimizer.

## Hardware
-----------
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz
- Memory: 32G
- NVIDIA GeForce RTX 2080 * 1

## Pipeline
-----------
To reproduce the result, there are three steps:
1. Dataset Preparation
2. Training / Testing
3. Inference

## Environment
-----------
### Docker Env
```
docker build -t "protcnn" .

docker run -it --gpus all --network host -v [local dir]:[docker dir] --shm-size=32G -d --name [container name] protcnn

// Execute container (Optional)
docker exec -it [containerID] bash
```
Project directory structured
```
+-- assets/
|   +-- family_distribution.png
|   +-- ...
+-- backbone/
|   +-- layers/
|   |   +--  residual_block.py
|   |   +--  ...
|   +-- resnet.py
+-- output/
|   +-- exp1
|   +-- ...
+-- dataset/
|   +-- train
|   +-- dev
|   +-- test
+-- utils/
|   +-- analysis.py
|   +-- dataloader.py
|   +-- metrics.py
|   +-- model.py
|   +-- optimizer.py
|   +-- tools.py
+-- Dockerfile
+-- README.md
+-- requirements.txt
+-- train.py
+-- test.py
```

## Dataset preparation
---------------

### Download Dataset
["Kaggle: Pfam seed random split"](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split)

### Extract random_split.zip to dataset directory.
After downloading and pre-process, the data directory is structured as:
```
+-- dataset/
|   +-- train
|   +-- |   +-- data-00000-of-00080
|   +-- |   +-- ...
|   +-- dev
|   +-- |   +-- data-00000-of-00010
|   +-- |   +-- ...
|   +-- test
|   +-- |   +-- data-00000-of-00010
|   +-- |   +-- ...
```
## Dataset Analysis
---------------
```
python utils/analysis.py
```

## Training
---------------
### Usage

```
usage: train.py [-h] [--output_version OUTPUT_VERSION] [--data_dir DATA_DIR]
                [--train_dir TRAIN_DIR] [--val_dir VAL_DIR]
                [--backbone BACKBONE] [--snapshot SNAPSHOT]
                [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--optimizer OPTIMIZER] [--lr LR] [--early_stop EARLY_STOP]
                [--scheduler SCHEDULER] [--steps_per_shot STEPS_PER_SHOT]
                [--save_checkpoint SAVE_CHECKPOINT]
                [--seq_max_len SEQ_MAX_LEN] [--weight_decay WEIGHT_DECAY]
                [--loss_function LOSS_FUNCTION] [--num_workers NUM_WORKERS]
                [--seed SEED]
```

### Example Usage

Train resnet

```python
python train.py --output_version resnet --data_dir ./random_split --train_dir train --val_dir val --num_epochs 10 --batch_size 256 \
        --backbone resSEnet --optimizer RAdam --early_stop False --lr 0.005 --weight_decay 0 --loss_function categorical --num_workers 8 
```

Train senet

```python
python train.py --output_version senet --data_dir ./random_split --train_dir train --val_dir val --num_epochs 10 --batch_size 256 \
        --backbone senet --optimizer RAdam --early_stop False --lr 0.005 --weight_decay 0 --loss_function categorical --num_workers 8 
```

Visualize

```
tensorboard --logdir "./output/writer"
```
## Evaluate (Test) 
---------------
### Usage

```
usage: test.py [-h] --model MODEL [--backbone BACKBONE]
               [--save_result SAVE_RESULT] [--data_dir DATA_DIR]
               [--train_dir TRAIN_DIR] [--test_dir TEST_DIR]
               [--batch_size BATCH_SIZE] [--seq_max_len SEQ_MAX_LEN]
               [--use_gpu USE_GPU] [--gpu GPU] [--num_workers NUM_WORKERS]
               [--seed SEED]
```

### Example Usage

test trained model

```python
python test.py --model /home/workspace/protCNN/output/snapshots/seNet_test/epoch_8iter_1001.pth --backbone resnet --data_dir ./random_split --train_dir train --test_dir test --batch_size 256 --seq_max_len 120 --num_workers 8 
```

## Result (with adam optimizer)
---------------

|       |Params|  Top1  | Top 5 |
|:-----:|:----:| :----: | :----:|
| resnet |137M| 0.9668  | 0.9827 |
| resSEnet |137M| 0.9668  | 0.9830 |
| mobilenetV2 |18 M| 0.9504  | 0.9711 |

## Reference
---------------
- [Using Deep Learning to Annotate the Protein Universe](https://www.biorxiv.org/content/10.1101/626507v3.full.pdf)
- [Kaggle: Pfam seed random split](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split)