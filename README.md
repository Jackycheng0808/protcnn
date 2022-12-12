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
+-- backbone/
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

## Training and Testing
---------------
### Train Model 

```
./train_sh/train.sh
```

### Evaluation Model
```
./test_sh/test.sh
```

## Result
---------------

## Reference
---------------
- [Using Deep Learning to Annotate the Protein Universe](https://www.biorxiv.org/content/10.1101/626507v3.full.pdf)
- [Kaggle: Pfam seed random split](https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split)