## Installation
### Python 3.8
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### Install Pytorch
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Install MMEngine and MMCV using MIM
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
```

### Install mmdet
```bash
mim install "mmdet>=3.1.0"
```

### Install mmpose
```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

### Excecute Check
```bash
python3 version_check.py
```

## Demo
### Download pretrained model
```bash
mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192 --dest hrnet
```

### Run demo
```bash
python3 mmpose/demo/image_demo.py \
    examples/img1.jpg \
    hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file examples/img1_result.jpg \
    --draw-heatmap
```

```bash
python3 mmpose/demo/image_demo.py \
    examples/img2.png \
    hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file examples/img2_result.jpg \
    --draw-heatmap
```

### Run Python Script
```bash
Python3 first_demo.py
```

## RTM Demo
### Download RTM model
```bash
mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192 --dest checkpoints
mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest checkpoints
```

### Run RTM Python Script
```bash
python3 second_demo.py -i examples/img1.jpg
python3 second_demo.py -i examples/img2.png
```

## MMPose Tutorial
- [Colab](https://github.com/open-mmlab/mmpose/blob/main/demo/MMPose_Tutorial.ipynb) :  Demo program file for using MMPose in Colab
- [A 20-MINUTE TOUR TO MMPOSE](https://mmpose.readthedocs.io/en/latest/guide_to_framework.html#a-20-minute-tour-to-mmpose)

## Reference
- [mmpose](https://mmpose.readthedocs.io/en/latest/)
- [mmdet](https://mmdetection.readthedocs.io/en/latest/)
- [Hrnet on AnimalPose](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html#topdown-heatmap-hrnet-on-animalpose)