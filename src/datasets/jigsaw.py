"""
This code is obtained from https://github.com/Felix-Petersen/diffsort and then modified.
"""

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from einops import rearrange


class JigsawDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        num_compare,
        seed=0,
        determinism=True,
    ):
        super().__init__()

        self.images = images
        self.labels = labels
        self.num_compare = num_compare
        self.seed = seed
        self.rand_state = None

        self.determinism = determinism

        if determinism:
            self.reset_rand_state()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):

        if self.determinism:
            prev_state = torch.random.get_rng_state()
            torch.random.set_rng_state(self.rand_state)

        id = torch.randint(len(self), ())
        label = self.labels[id]

        img = self.images[id].type(torch.float32) / 255.

        h1 = img.shape[1] // self.num_compare
        w1 = img.shape[2] // self.num_compare
        assert h1 == w1

        pieces = rearrange(img, 'c (h1 h) (w1 w) -> (h1 w1) c h w', h1=self.num_compare, w1=self.num_compare)
        indices = torch.randperm(pieces.shape[0])
        pieces = pieces[indices]

        if self.determinism:
            self.rand_state = torch.random.get_rng_state()
            torch.random.set_rng_state(prev_state)

        return pieces, indices, img, label

    def reset_rand_state(self):
        prev_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed)
        self.rand_state = torch.random.get_rng_state()
        torch.random.set_rng_state(prev_state)


class JigsawSplits:
    def __init__(self, dataset, num_compare, seed=0, deterministic_data_loader=True):
        self.deterministic_data_loader = deterministic_data_loader

        if dataset == 'mnist':
            trva_real = datasets.MNIST(root='./data-mnist', download=True)
            xtr_real = trva_real.data[:55000].view(-1, 1, 28, 28)
            ytr_real = trva_real.targets[:55000]
            xva_real = trva_real.data[55000:].view(-1, 1, 28, 28)
            yva_real = trva_real.targets[55000:]

            te_real = datasets.MNIST(root='./data-mnist', train=False, download=True)
            xte_real = te_real.data.view(-1, 1, 28, 28)
            yte_real = te_real.targets

            if num_compare == 3:
                xtr_real = xtr_real[:, :, :27, :27]
                xva_real = xva_real[:, :, :27, :27]
                xte_real = xte_real[:, :, :27, :27]

            self.train_dataset = JigsawDataset(
                images=xtr_real, labels=ytr_real, num_compare=num_compare, seed=seed,
                determinism=deterministic_data_loader)
            self.valid_dataset = JigsawDataset(
                images=xva_real, labels=yva_real, num_compare=num_compare, seed=seed)
            self.test_dataset = JigsawDataset(
                images=xte_real, labels=yte_real, num_compare=num_compare, seed=seed)

            assert num_compare in [2, 3, 4, 7, 14, 28]
        elif dataset == 'cifar10':
            trva_real = datasets.CIFAR10(root='./data-cifar10', download=True, train=True)
            xtr_real = torch.tensor(trva_real.data[:45000]).permute(0, 3, 1, 2)
            ytr_real = torch.tensor(trva_real.targets[:45000])

            xva_real = torch.tensor(trva_real.data[45000:]).permute(0, 3, 1, 2)
            yva_real = torch.tensor(trva_real.targets[45000:])

            te_real = datasets.CIFAR10(root='./data-cifar10', download=True, train=False)
            xte_real = torch.tensor(te_real.data).permute(0, 3, 1, 2)
            yte_real = torch.tensor(te_real.targets)

            if num_compare == 3:
                xtr_real = xtr_real[:, :, 1:31, 1:31]
                xva_real = xva_real[:, :, 1:31, 1:31]
                xte_real = xte_real[:, :, 1:31, 1:31]

            self.train_dataset = JigsawDataset(
                images=xtr_real, labels=ytr_real, num_compare=num_compare, seed=seed,
                determinism=deterministic_data_loader)
            self.valid_dataset = JigsawDataset(
                images=xva_real, labels=yva_real, num_compare=num_compare, seed=seed)
            self.test_dataset = JigsawDataset(
                images=xte_real, labels=yte_real, num_compare=num_compare, seed=seed)

            assert num_compare in [2, 3, 4, 8, 16, 32]
        else:
            raise NotImplementedError()

    def get_train_loader(self, batch_size, **kwargs):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  num_workers=4 if not self.deterministic_data_loader else 0,
                                  shuffle=True, **kwargs)
        return train_loader

    def get_valid_loader(self, batch_size, **kwargs):
        valid_loader = DataLoader(self.valid_dataset,
                                  batch_size=batch_size, shuffle=False, **kwargs)
        return valid_loader

    def get_test_loader(self, batch_size, **kwargs):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=batch_size, shuffle=False, **kwargs)
        return test_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    splits = JigsawSplits(dataset='mnist', num_compare=2, seed=42)
    splits = JigsawSplits(dataset='cifar10', num_compare=2, seed=42)

    data_loader_train = splits.get_train_loader(batch_size=128, drop_last=True)

    for _ in range(0, 2):
        count = 0

        for pieces, indices, imgs, labels in data_loader_train:
            count += 1
            print(labels)
            print(indices[0])
            print(pieces.shape, indices.shape, imgs.shape, labels.shape)

            plt.imshow(imgs[0].permute(1, 2, 0))
            plt.show()

            for ind in range(0, pieces.shape[1]):
                piece = pieces[0, ind]
                print(piece.shape)

                plt.imshow(piece.permute(1, 2, 0))
                plt.show()

        print(count)
