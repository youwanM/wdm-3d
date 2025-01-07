# general settings
GPU=0;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=32;              # number of model base channels (we use 64 for all experiments)
MODE='training';           # train vs sample
DATASET='brats';          # brats or lidc-idri
MODEL='ours_wnet_256';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

# settings for sampling/inference
ITERATIONS=069;             # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0;         # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="runs/Jan06_16-47-57_ptb-03240161.irisa.fr";               # tensorboard dir to be set for the evaluation

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "MODEL: WDM (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=4;
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "MODEL: WDM (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "MODEL: WDM (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=4;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "MODEL: WDM (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

# some information and overwriting batch size for sampling
# (overwrite in case you want to sample with a higher batch size)
# no need to change for reproducing
if [[ $MODE == 'sample' ]]; then
  echo "MODE: sample"
  BATCH_SIZE=1
elif [[ $MODE == 'healthy' ]]; then
  DATA_DIR=~/Desktop/BRATS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
  echo "MODE: healthy"
  BATCH_SIZE=1
elif [[ $MODE == 'train' ]]; then
  if [[ $DATASET == 'brats' ]]; then
    echo "MODE: training"
    echo "DATASET: BRATS"
    DATA_DIR=~/Desktop/BRATS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
  elif [[ $DATASET == 'lidc-idri' ]]; then
    echo "MODE: training"
    echo "DATASET: LIDC-IDRI"
    DATA_DIR=~/wdm-3d/data/LIDC-IDRI/
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again"
    exit 1
  fi
else
  echo "MODE NOT FOUND -> Supported modes: sample, healthy, train"
  exit 1
fi


COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=8
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
"
TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=1000
--num_workers=24
--devices=${GPU}
"
SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=./${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--devices=${GPU}
--output_dir=./results/${RUN_DIR}/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
--num_workers=24
"

# Run the Python scripts
if [[ $MODE == 'train' ]]; then
  python scripts/generation_train.py $TRAIN $COMMON
elif [[ $MODE == 'sample' ]]; then
  python scripts/generation_sample.py $SAMPLE $COMMON
elif [[ $MODE == 'healthy' ]]; then
  python scripts/generation_healthy.py $SAMPLE $COMMON
else
  echo "ERROR: Unsupported MODE -> $MODE"
  exit 1
fi
