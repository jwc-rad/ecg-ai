# ecg-ai
This repository provides PyTorch implementation of deep learning model for estimating age from electrocardiogram (ECG). This was the winning solution for the [MAIC ECG AI 2023 challenge](https://maic.or.kr/competitions/26/infomation).

## Installation
Create a virtual environment and activate it.
```sh
conda create -n ecgai python=3.9
conda activate ecgai
```
Install required packages
```sh
git clone https://github.com/jwc-rad/MISLight.git
cd MISLight
pip install -e .

cd ..
git clone https://github.com/jwc-rad/ecg-ai.git
cd ecg-ai
pip install -r requirements.txt
```

## Dataset
The current configurations assume the input shape as `12x5000`, a 10-second 12-lead ECG with the sampling frequence of 500Hz.
There are some public ECG datasets with the same data format as follows:
- [A large scale 12-lead electrocardiogram database for arrhythmia study](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)

## Train
```
python train.py experiment=maic_v1 paths.data_root_dir=${PATH_TO_DATA_DIRECTORY}
```
- For more options, get started from `config/experiment/maic_v1.yaml`
