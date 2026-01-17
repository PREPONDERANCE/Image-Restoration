import numpy as np
import os
import argparse
from tqdm import tqdm
import math

import jittor as jt
import jittor.nn as nn
import utils

from natsort import natsorted
from glob import glob

from skimage.util import img_as_ubyte

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from basicsr.models import create_model
from basicsr.utils.options import parse

jt.flags.use_cuda = 1  # Use GPU

parser = argparse.ArgumentParser(description="Image Deraining using Restormer")

parser.add_argument(
    "--input_dir",
    default="/mnt/sda/zsh/derain/",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir", default="./results/FPro/", type=str, help="Directory for results"
)
parser.add_argument(
    "--weights", default="./models/derain_spad.pth", type=str, help="Path to weights"
)
parser.add_argument(
    "--opt",
    type=str,
    default="Options/Deraining_FPro_spad.yml",
    help="Path to option YAML file.",
)
parser.add_argument("--gpus", type=str, default="1", help="GPU devices.")

args = parser.parse_args()

gpu_list = ",".join(str(x) for x in args.gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
print("export CUDA_VISIBLE_DEVICES=" + gpu_list)


def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs : hs + crop_size, ws : ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = jt.ones((B, C, H, W))
    if not is_mean:
        score_np = np.ones((B, C, H, W))
        for h in range(H):
            for w in range(W):
                score_np[:, :, h, w] = 1.0 / (
                    math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6)
                )
        score = jt.array(score_np)
    return score


def mergeimage(split_data, starts, crop_size=128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = jt.zeros((B, C, H, W))
    merge_img = jt.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=True)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs : hs + crop_size, ws : ws + crop_size] += scoremap * simg
        tot_score[:, :, hs : hs + crop_size, ws : ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


####### Load yaml #######
yaml_file = args.opt
weights = args.weights


opt = parse(args.opt, is_train=False)
opt["dist"] = False


x = yaml.load(open(args.opt, mode="r"), Loader=Loader)
s = x["network_g"].pop("type")
##########################

# IMPORTANT: The FPro class must be implemented using Jittor's nn.Module
model_restoration = create_model(opt).net_g
checkpoint = jt.load(args.weights)

if s == "HINT":
    del checkpoint["params"]["qv_cache"]

model_restoration.load_state_dict(checkpoint["params"])
print("===>Testing using weights: ", args.weights)
model_restoration.eval()


factor = 8
datasets = ["real_test_1000"]

for dataset in datasets:
    result_dir = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    # inp_dir = os.path.join(args.input_dir, 'test', dataset, 'rain')
    inp_dir = os.path.join(args.input_dir, dataset, "rain")
    files = natsorted(
        glob(os.path.join(inp_dir, "*.png")) + glob(os.path.join(inp_dir, "*.jpg"))
    )
    with jt.no_grad():
        for file_ in tqdm(files):
            img = np.float32(utils.load_img(file_)) / 255.0

            input_ = jt.array(img).permute(2, 0, 1)
            input_ = input_.unsqueeze(0)

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0

            input_ = nn.pad(input_, (0, padw, 0, padh), "reflect")

            B, C, H, W = input_.shape
            corp_size_arg = 256
            overlap_size_arg = 158

            split_data, starts = splitimage(
                input_, crop_size=corp_size_arg, overlap_size=overlap_size_arg
            )
            for i, data in enumerate(split_data):
                # Jittor change: model execution returns a Jittor Var, no need for .cpu()
                split_data[i] = model_restoration(data)
            restored = mergeimage(
                split_data, starts, crop_size=corp_size_arg, resolution=(B, C, H, W)
            )

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            # Jittor change: clamp, permute, squeeze are similar. .numpy() instead of .detach().cpu().numpy()
            restored = restored.clamp(0, 1).permute(0, 2, 3, 1).squeeze(0).numpy()

            utils.save_img(
                (
                    os.path.join(
                        result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + ".png"
                    )
                ),
                img_as_ubyte(restored),
            )
