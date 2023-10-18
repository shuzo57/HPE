# HPE (Human Pose Estimation) using MMPose
## Environment (Kitano)
`version_check.py` で確認できます。
- Python version: 3.8.18
- torch version: 1.13.1+cu117 True
- torchvision version: 0.14.1+cu117
- mmpose version: 1.2.0
- cuda version: 11.7
- compiler information: GCC 9.3

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

### Run Python Script
- `first_demo.py` は、姿勢推定（Pose Estimation）のデモプログラムで、指定した画像に対して姿勢推定を行います。プログラムは、OpenMMLabの`mmpose`ライブラリを使用してモデルを初期化し、画像から姿勢情報を抽出し、結果を表示します。デモの目的は、姿勢推定モデルの動作を示すことです。
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
- `second_demo.py` は、物体検出と姿勢推定を結合したデモプログラムで、指定した入力画像に対して人物の姿勢を検出し、結果を可視化します。このプログラムは、OpenMMLabの`mmdet`と`mmpose`ライブラリを活用しています。
```bash
python3 second_demo.py -i examples/img1.jpg
```

## MMPose Tutorial
- [Colab](https://github.com/open-mmlab/mmpose/blob/main/demo/MMPose_Tutorial.ipynb) :  Demo program file for using MMPose in Colab
- [A 20-MINUTE TOUR TO MMPOSE](https://mmpose.readthedocs.io/en/latest/guide_to_framework.html#a-20-minute-tour-to-mmpose)

## Reference
- [mmpose](https://mmpose.readthedocs.io/en/latest/)
- [mmdet](https://mmdetection.readthedocs.io/en/latest/)
- [Hrnet on AnimalPose](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html#topdown-heatmap-hrnet-on-animalpose)

# SkeletonPose
## Installation
```bash
pip install -e SkeletonPose
```

## Demo
M model
```bash
python3 -m skeletonpose \
    -i examples/video1.mp4 \
    -o test \
    -pc checkpoints/rtmpose-m_8xb256-420e_coco-256x192.py \
    -pck checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth \
    -dc checkpoints/rtmdet_m_8xb32-300e_coco.py \
    -dck checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth
```

L model
```bash
python3 -m skeletonpose \
    -i examples/video1.mp4 \
    -o test \
    -pc checkpoints/rtmpose-l_8xb256-420e_coco-256x192.py \
    -pck checkpoints/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth \
    -dc checkpoints/rtmdet_l_8xb32-300e_coco.py \
    -dck checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
```