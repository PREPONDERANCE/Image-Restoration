import numpy as np
import os
import math
import argparse
from tqdm import tqdm

import jittor as jt
import jittor.nn as nn


from skimage.util import img_as_ubyte
from natsort import natsorted
from glob import glob
from Denoising import utils

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from basicsr.models import create_model
from basicsr.utils.options import parse


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
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (
                    math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6)
                )
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


parser = argparse.ArgumentParser(description="Gaussian Color Denoising using HINT")

parser.add_argument(
    "--input_dir",
    default="./Denoising/Datasets/test/",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir",
    default="./results/Gaussian_Color_Denoising/",
    type=str,
    help="Directory for results",
)
parser.add_argument(
    "--opt",
    type=str,
    default="./Denoising/Options/GaussianColorDenoising_HINT.yml",
    help="Path to option YAML file.",
)
parser.add_argument(
    "--weights", default="./models/net_g_latest", type=str, help="Path to weights"
)
parser.add_argument("--gpus", type=str, default="1", help="GPU devices.")
# parser.add_argument(
#     "--model_type",
#     required=True,
#     choices=["non_blind", "blind"],
#     type=str,
#     help="blind: single model to handle various noise levels. non_blind: separate model for each noise level.",
# )
parser.add_argument("--sigmas", default="15,25,50", type=str, help="Sigma values")

args = parser.parse_args()

gpu_list = ",".join(str(x) for x in args.gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

opt = parse(args.opt, is_train=False)
opt["dist"] = False

####### Load yaml #######
# if args.model_type == "blind":
#     yaml_file = "Options/GaussianColorDenoising_HINT.yml"
# else:
#     yaml_file = f"Options/GaussianColorDenoising_RestormerSigma{args.sigmas}.yml"

yaml_file = args.opt
with open(yaml_file, "r") as f:
    x = yaml.load(f, loader=Loader)

s = x["network_g"].pop("type")
##########################

sigmas = np.int_(args.sigmas.split(","))
factor = 8
datasets = ["CBSD68", "Urban100"]

for sigma_test in sigmas:
    print("Compute results for noise level", sigma_test)
    model_restoration = create_model(opt).net_g
    # if args.model_type == "blind":
    #     weights = args.weights + "_blind.jth"
    # else:
    #     weights = args.weights + "_sigma" + str(sigma_test) + ".jth"
    checkpoint = jt.load(args.weights)
    if s == "HINT":
        del checkpoint["params"]["qv_cache"]
    model_restoration.load_state_dict(checkpoint["params"])

    print("===>Testing using weights: ", args.weights)
    print("------------------------------------------------")
    model_restoration.cuda()
    model_restoration.eval()

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, dataset)
        files = natsorted(
            glob(os.path.join(inp_dir, "*.png")) + glob(os.path.join(inp_dir, "*.tif"))
        )
        result_dir_tmp = os.path.join(args.result_dir, dataset, str(sigma_test))
        os.makedirs(result_dir_tmp, exist_ok=True)

        with jt.no_grad():
            for file_ in tqdm(files):
                img = np.float32(utils.load_img(file_)) / 255.0

                np.random.seed(seed=0)  # for reproducibility
                img += np.random.normal(0, sigma_test / 255.0, img.shape)

                img = jt.array(img).permute(2, 0, 1)
                input_ = img.unsqueeze(0).cuda()

                # Padding in case images are not multiples of 8
                h, w = input_.shape[2], input_.shape[3]
                H, W = (
                    ((h + factor) // factor) * factor,
                    ((w + factor) // factor) * factor,
                )
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                input_ = nn.pad(input_, (0, padw, 0, padh), "reflect")

                if s in {"HINT", "AST", "ASTv2"}:
                    restored = model_restoration(input_)
                elif s == "FPro":
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
                        split_data,
                        starts,
                        crop_size=corp_size_arg,
                        resolution=(B, C, H, W),
                    )

                # Unpad images to original dimensions
                restored = restored[:, :, :h, :w]

                restored = (
                    jt.clamp(restored, 0, 1)
                    .cpu()
                    .detach()
                    .permute(0, 2, 3, 1)
                    .squeeze(0)
                    .numpy()
                )

                save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                utils.save_img(save_file, img_as_ubyte(restored))
