import pickle

import einops
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torchvision import transforms as T, utils
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T, utils


class Identity:
    def __call__(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Cifar10Dataset(Dataset):
    def __init__(
            self,
            cifar_10_batch_dir,
            image_size,
            augment_horizontal_flip=False,
            split='train'
    ):
        super().__init__()
        self.root_dir = cifar_10_batch_dir
        self.image_size = image_size
        self.image_shape = (32, 32, 3)  # cifar10 image shape
        self.num_classes = 10  # cifar10 has 10 classes

        # load the data batches and labels
        self.data, self.labels = self.load_cifar10_batches(split)

        # raise warning if image_size is to big as the images are 32x32
        if image_size > 32:
            print('Warning: image size is bigger than 32x32. Images will interpolated to the given size.')

        self.transform = T.Compose([
            T.Lambda(lambda x: einops.rearrange(x, 'c h w -> h w c')),
            T.ToPILImage(),
            T.Resize(self.image_size) if self.image_size != 32 else Identity(),
            T.RandomHorizontalFlip(p=1.0) if augment_horizontal_flip else Identity(),
            # T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.2) if split == 'train' else Identity(),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]  # [3, 32, 32]
        label = self.labels[index]
        return self.transform(img)  # , label  # [3, 32, 32]

    def load_cifar10_batches(self, split):
        if split == 'train':
            batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        elif split == 'test':
            batches = ['test_batch']
        else:
            raise ValueError("split parameter must be either 'train' or 'test'.")

        data = []
        labels = []

        # load data batches
        for batch in batches:
            with open(f'{self.root_dir}/{batch}', 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
                data.append(batch_data[b'data'])
                labels += batch_data[b'labels']

        # concatenate the data batches and reshape to the cifar10 image shape
        # data -- a 50000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        data = np.concatenate(data, axis=0)
        data = data.reshape((data.shape[0], 3, 32, 32))

        return data, labels


if __name__ == "__main__":
    from pathlib import Path

    env_file = Path(__file__).parent.parent / "env.yaml"
    env = yaml.safe_load(env_file.open())
    dataset = Cifar10Dataset(
        cifar_10_batch_dir=Path(env["dataset_dir"]) / "cifar-10-batches-py",
        image_size=32,
        augment_horizontal_flip=True,
        convert_image_to=None,
        split='train'
    )

    print(dataset[0][0].shape)
    print(dataset[0][1])
