import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th

sys.path.append(".")

from guided_diffusion import (dist_util, logger)
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion, add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()

def add_noise(img, diffusion, steps):
    for step in range(steps):
        img = diffusion.q_sample(img, th.tensor([step] * img.shape[0], device=img.device))
    return img

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    # Load input image
    args.input_image = "/home/ymahe/Desktop/BRATS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz"
    input_image_path = args.input_image
    input_image = nib.load(input_image_path).get_fdata()
    input_image = th.tensor(input_image, dtype=th.float32).unsqueeze(0).unsqueeze(0).to(dist_util.dev())


    # Apply DWT to the input image
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(input_image)
    print(LLL.shape)
    print(LLH.shape)
    print(LHL.shape)
    print(LHH.shape)

    dwt_image = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)


    # Add noise to the input image
    noisy_image = add_noise(dwt_image, diffusion, args.noise_steps)

    for ind in range(args.num_samples // args.batch_size):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        seed += 1

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model, shape=noisy_image.shape, noise=noisy_image)

        B, _, D, H, W = sample.size()

        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample = (sample + 1) / 2.

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)

        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(sample.shape[0]):
            output_name = os.path.join(args.output_dir, f'sample_{ind}_{i}.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,
        num_workers=24,
        input_image="",
        noise_steps=500,
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
"""
A script for sampling from a diffusion model for unconditional image generation.


import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          add_dict_to_argparser,
                                          args_to_dict,
                                          )
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices


    ds = BRATSVolumes(args.data_dir, test_flag=False,
                      normalize=(lambda x: 2*x - 1) if args.renormalize else None,
                      mode='train',
                      img_size=args.image_size)
    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=True,
                                     )
    cuda = th.device("cuda:0")
    first_batch = next(iter(datal)).to(cuda)

    model.eval()
    dwt = DWT_3D("haar")
    idwt = IDWT_3D("haar")

    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(first_batch)
    x_start_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

    if args.use_fp16:
        raise ValueError("fp16 currently not implemented")

    for ind in range(args.num_samples // args.batch_size):
        th.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # print(f"Reseeded (in for loop) to {seed}")

        seed += 1

        img = th.randn(args.batch_size,         # Batch size
                       8,                       # 8 wavelet coefficients
                       args.image_size//2,      # Half spatial resolution (D)
                       args.image_size//2,      # Half spatial resolution (H)
                       args.image_size//2,      # Half spatial resolution (W)
                       ).to(dist_util.dev())

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop_known

        sample = sample_fn(model=model,
                           shape=img.shape,
                           img=img,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs,
                           )

        B, _, D, H, W = sample.size()

        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample = (sample + 1) / 2.


        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for i in range(sample.shape[0]):
            output_name = os.path.join(args.output_dir, f'sample_{ind}_{i}.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='../results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        num_workers = 24,

    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
"""