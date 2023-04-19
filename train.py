import os
from pathlib import Path
import torch

from dataset.cifar10 import Cifar10Dataset
from rin_pytorch import GaussianDiffusion, RIN, Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("WARNING: Training on CPU is not recommended. Consider using a GPU.")
if "cuda" in str(device):
    print(f"Found GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

model = RIN(
    dim=128,  # model dimensions
    image_size=32,  # image size
    patch_size=4,  # patch size
    depth=6,  # depth
    num_latents=128,  # number of latents. they used 256 in the paper
    dim_latent=256,  # can be greater than the image dimension (dim) for greater capacity
    latent_self_attn_depth=4,  # number of latent self attention blocks per recurrent step, K in the paper
).to(device)


#print model parameters as XMillion
print("Total number of parameters: {:.2f} Million".format(sum(p.numel() for p in model.parameters()) / 1e6))

# check if
if "cuda" in str(device):
    if torch.cuda.get_device_capability(0)[0] >= 7:
        print("Compiling model for CUDA... (this may take a while)")
        model = torch.compile(model)
    else:
        print("WARNING: CUDA Capability < 7.0. Compiling model for CUDA is not supported.")

diffusion = GaussianDiffusion(
    model,
    timesteps=400,
    train_prob_self_cond=0.9,  # how often to self condition on latents
    scale=1.
    # this will be set to < 1. for more noising and leads to better convergence when training on higher resolution images (512, 1024) - input noised images will be auto variance normalized
).to(device)

dataset = Cifar10Dataset(
    cifar_10_batch_dir=Path(os.environ["DATASET_DIR"]) / "cifar-10-batches-py",
    image_size=32,
    augment_horizontal_flip=True
)

# folderDataset = FolderImageDataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)

trainer = Trainer(
    diffusion,
    dataset=dataset,
    num_samples=16,
    train_batch_size=32,
    gradient_accumulate_every=1,
    train_lr=1e-4,
    save_and_sample_every=1000,
    train_num_steps=100_000,  # total training steps
    ema_decay=0.995,  # exponential moving average decay
    fp16=True,
    wandb_project_name="rin-pytorch",
    wandb_run_name="cifar10",
)

if __name__ == '__main__':
    # load last checkpoint if exists
    milestones = [int(str(f.stem).split("-")[1]) for f in trainer.results_folder.glob("model-*.pt")]
    if len(milestones) > 0:
        trainer.load(max(milestones))

    trainer.train()
