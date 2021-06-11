
## 0. Create env

```bash
# create env
conda create --name sandbox --clone base
# activate
source activate sandbox

# installing required packages
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia --yes
pip install -U catalyst==21.05 \
    pytorch-lightning==1.3.5 \
    pytorch-ignite==0.4.4 \
    pynvml==11.0.0 \
    albumentations==1.0.0
```


## 1. Catalyst Python API

```bash
python catalyst-python-api/experiment.py
```


## 2. Catalyst Config API

```bash
catalyst-dl run \
    --expdir=./catalyst-config-api \
    --config=./catalyst-config-api/config.yaml \
    --logdir=./logs/catalyst-config-api
```

feature of Config API:

```bash
catalyst-dl run \
    --expdir=./catalyst-config-api \
    --config=./catalyst-config-api/config.yaml \
    --logdir=./logs/catalyst-config-api \
    --stages/stage_0/num_epochs=5:int
```


## 3. PyTorch Lightning

```bash
python lightning/experiment.py \
    --default_root_dir='logs/lightning' \
    --gpus=1 \
    --max_epochs=3
```


## 4. PyTorch Ignite

```bash
python ignite/experiment.py
```
