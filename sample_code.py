import torch

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
register_all_modules()

config_file = "hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
checkpoint_file = (
    "hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"
)
model = init_model(config_file, checkpoint_file, device=device)

results = inference_topdown(model, "examples/img1.jpg")
