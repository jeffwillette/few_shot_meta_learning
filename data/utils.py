from argparse import Namespace
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset  # type: ignore

from data import (MetaCircles, MetaMoons, MiniImageNet,
                  MiniImageNetCorruptTest, Omniglot, OmniglotCorruptTest,
                  RandomGaussians)

corruptions = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "pixelate",
    "saturate",
    "shot_noise",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]


def get_random_corruptions(train_n: int, test_n: int) -> Tuple[List[str], ...]:
    if train_n + test_n != len(corruptions):
        raise ValueError(f"train_n and test_n need to equal corruptions len: {train_n + test_n}")

    idx = torch.randperm(len(corruptions))
    train_c = [v for (i, v) in enumerate(corruptions) if i in idx[:train_n].tolist()]
    test_c = [v for (i, v) in enumerate(corruptions) if i in idx[train_n:].tolist()]
    return train_c, test_c


def get_dataset(args: Namespace, root: str, name: str, batch_size: int, num_workers: int, ood_test: bool = False, seed: int = 0) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    train: Dataset
    test: Dataset
    train_ldr: DataLoader
    test_ldr: DataLoader

    if name == "omniglot":
        train = Omniglot(root, split="train", n_way=args.n_way, k_shot=args.k_shot, test_shots=args.test_shots, _len=args.batch_size * 100)
        test = Omniglot(root, split="test", n_way=args.n_way, k_shot=args.k_shot, test_shots=args.test_shots, ood_test=ood_test)
        train_ldr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_ldr = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_ldr, None, test_ldr
    elif name == "miniimagenet":
        train = MiniImageNet(root, split="train", n_way=args.n_way, k_shot=args.k_shot, test_shots=args.test_shots, _len=args.batch_size * 100)
        val = MiniImageNet(root, split="val", n_way=args.n_way, k_shot=args.k_shot, test_shots=args.test_shots, _len=args.batch_size * 100)
        test = MiniImageNet(root, split="test", n_way=args.n_way, k_shot=args.k_shot, test_shots=args.test_shots, ood_test=ood_test)

        train_ldr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_ldr = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_ldr = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_ldr, val_ldr, test_ldr
    elif name == "miniimagenet-meta-test-corrupt":
        train = MiniImageNet(root, split="train", n_way=args.n_way, k_shot=args.k_shot, test_shots=args.test_shots, _len=args.batch_size * 100)
        test = MiniImageNetCorruptTest(root, split="test", n_way=args.n_way, k_shot=args.k_shot, test_shots=90)

        train_ldr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_ldr = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_ldr, None, test_ldr
    elif name == "omniglot-meta-test-corrupt":
        # meta train set is the standard omniglot training set, we increase the test set size by a factor of 17 to account for all of the different corruptions
        train = Omniglot(root, split="train", n_way=args.n_way, k_shot=args.k_shot, test_shots=args.test_shots, _len=args.batch_size * 100)
        train_ldr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        # meta test set has a normal training set and a shifted test set
        test = OmniglotCorruptTest(root, split="test", n_way=args.n_way, k_shot=args.k_shot, test_shots=90)  # type: ignore
        test_ldr = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_ldr, None, test_ldr

    elif name == "toy-gaussian":
        train, test = RandomGaussians(n_way=args.n_way, k_shot=args.k_shot, test_shots=15, total_tasks=100, seed=seed), \
            RandomGaussians(n_way=args.n_way, k_shot=args.k_shot, total_tasks=5, seed=seed)

        train_ldr = DataLoader(train, batch_size=batch_size)
        test_ldr = DataLoader(test, batch_size=batch_size)
        return train_ldr, None, test_ldr
    elif name == "toy-moons":
        train, test = MetaMoons(k_shot=args.k_shot, test_shots=15, total_tasks=100, seed=seed), MetaMoons(k_shot=args.k_shot, total_tasks=5, seed=seed)
        train_ldr = DataLoader(train, batch_size=batch_size)
        test_ldr = DataLoader(test, batch_size=batch_size)
        return train_ldr, None, test_ldr
    elif name == "toy-circles":
        train, test = MetaCircles(k_shot=args.k_shot, test_shots=15, total_tasks=100, seed=seed), MetaCircles(k_shot=args.k_shot, total_tasks=5, seed=seed)
        train_ldr = DataLoader(train, batch_size=batch_size)
        test_ldr = DataLoader(test, batch_size=batch_size)
        return train_ldr, None, test_ldr
    else:
        raise NotImplementedError(f"dataset: {name} is not implemented")
