import os
from typing import Any, List, Tuple, Union

import numpy as np  # type: ignore
import torch
import torchvision  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CORRUPTIONS = [
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
    if train_n + test_n != len(CORRUPTIONS):
        raise ValueError(f"train_n and test_n need to equal corruptions len: {train_n + test_n}")

    idx = np.random.permutation(len(CORRUPTIONS))
    train_c = [v for (i, v) in enumerate(CORRUPTIONS) if i in idx[:train_n].tolist()]
    test_c = [v for (i, v) in enumerate(CORRUPTIONS) if i in idx[train_n:].tolist()]
    return train_c, test_c


T = torch.Tensor
LT = Union[List, T]
transforms.RandomRotation


class NumpyDataset(Dataset):
    def __init__(self, root: str, split: str, n_way: int = 5, k_shot: int = 5, test_shots: int = 15, val_shots: int = 0, ood_test: bool = False, _len: int = None):
        """ood_test: if set to true, the test sets have random OOD classes (not seen during training)"""
        self.root = root
        self.split = split
        self.n_way = n_way
        self.k_shot = k_shot
        self.test_shots = test_shots
        self.val_shots = val_shots
        self.ood_test = ood_test
        self.len = _len

        self.data: Any
        self.dim: int
        self.ch: int
        self.n_classes: int
        self.n_per_class: int
        self.test_episodes: int

    def __len__(self) -> int:
        """
        this is not really important for the training set since every task is sampled randomly, and we measure total tasks not epochs.
        For training, this is intended to be set to a round number to decide when to do validation.

        For testing, this should be some data length that is standard for the given test set.. Using values from prototypical networks
        """
        if self.split == "train":
            return self.len  # type: ignore
        elif self.split == "val":
            return self.n_classes
        elif self.split == "test":
            return self.test_episodes
        else:
            raise NotImplementedError("split not implemented")

    def random_classes_and_shots(self, n_way: int, k_shot: int, test_shots: int) -> Tuple[Any, ...]:
        classes = np.random.choice(self.n_classes, size=n_way, replace=False)
        return classes, np.random.permutation(self.n_per_class)

    def get_empty(self) -> Tuple[T, T, T, T]:
        return (
            torch.zeros(self.n_way * self.k_shot, self.ch, self.dim, self.dim),
            torch.zeros(self.n_way * self.k_shot),
            torch.zeros(self.n_way * self.test_shots, self.ch, self.dim, self.dim),
            torch.zeros(self.n_way * self.test_shots),
        )

    def getitem_ood_test(self, i: int) -> Tuple[T, T, T, T]:
        spt_x, spt_y, qry_x, qry_y = self.get_empty()

        classes, idx = self.random_classes_and_shots(self.n_way * 2, self.k_shot, self.test_shots)
        train_idx, test_idx = idx[:self.k_shot], idx[self.k_shot : self.k_shot + self.test_shots]

        # get a test set with random classes in it
        for i, cl in enumerate(classes[:self.n_way]):  # get the first n classes
            spt_x[(i * self.k_shot) : ((i + 1) * self.k_shot)] = self.data[cl, train_idx]
            spt_y[(i * self.k_shot) : ((i + 1) * self.k_shot)] = i

        for i, cl in enumerate(classes[self.n_way : self.n_way * 2]):  # get a different set of classes for ood
            qry_x[(i * self.test_shots) : ((i + 1) * self.test_shots)] = self.data[cl, test_idx]
            qry_y[(i * self.test_shots) : ((i + 1) * self.test_shots)] = i

        return spt_x, spt_y.long(), qry_x, qry_y.long()

    def getitem(self, i: int) -> Tuple[T, T, T, T]:
        spt_x, spt_y, qry_x, qry_y = self.get_empty()

        classes, idx = self.random_classes_and_shots(self.n_way, self.k_shot, self.test_shots)
        train_idx, test_idx = idx[:self.k_shot], idx[self.k_shot : self.k_shot + self.test_shots]
        # get the regular test set
        for i, cl in enumerate(classes):
            spt_x[(i * self.k_shot) : ((i + 1) * self.k_shot)] = self.data[cl, train_idx]
            spt_y[(i * self.k_shot) : ((i + 1) * self.k_shot)] = i

            qry_x[(i * self.test_shots) : ((i + 1) * self.test_shots)] = self.data[cl, test_idx]
            qry_y[(i * self.test_shots) : ((i + 1) * self.test_shots)] = i

        return spt_x, spt_y.long(), qry_x, qry_y.long()

    def generate_episode(self, episode_name: str = None) -> Tuple[T, T, T, T]:
        return self.__getitem__(0)  # index doesn't really matter

    def __getitem__(self, i: int) -> Tuple[T, T, T, T]:
        if self.ood_test:
            return self.getitem_ood_test(i)
        return self.getitem(i)


class Omniglot(NumpyDataset):
    def __init__(self, root: str, split: str, n_way: int = 5, k_shot: int = 5, test_shots: int = 15, val_shots: int = 0, ood_test: bool = False, _len: int = None):
        super().__init__(root, split, n_way, k_shot, test_shots, val_shots, ood_test, _len)

        # these will always be the test indices in the image folders
        self.path = os.path.join("omniglot", f"{n_way}-way", f"{k_shot}-shot", f"{test_shots}-testshot", f"{val_shots}-valshots")
        file = f"hb/omni_{split}_rot.npy"
        self.data = torch.from_numpy(np.load(os.path.join(root, "omniglot-resized-vinyals", file)))
        self.data = 1 - np.transpose(self.data, (0, 1, 4, 2, 3))

        self.dim = 28
        self.ch = 1
        self.n_classes = self.data.shape[0]
        self.n_per_class = 20
        self.test_episodes = 1000


class MiniImageNet(NumpyDataset):
    def __init__(self, root: str, split: str, n_way: int = 5, k_shot: int = 5, test_shots: int = 15, val_shots: int = 0, ood_test: bool = False, _len: int = None):
        """ood_test: if set to true, the test sets have random OOD classes (not seen during training)"""
        super().__init__(root, split, n_way, k_shot, test_shots, val_shots, ood_test, _len)

        # these will always be the test indices in the image folders
        self.path = os.path.join("miniimagenet", f"{n_way}-way", f"{k_shot}-shot", f"{test_shots}-testshot", f"{val_shots}-valshots")
        self.data = torch.from_numpy(np.load(os.path.join(root, "mimgnet", f"{split}.npy")))
        self.data = self.data.view(-1, 600, 84, 84, 3)
        self.data = np.transpose(self.data, (0, 1, 4, 2, 3))
        self.dim = 84
        self.ch = 3
        self.n_classes = self.data.shape[0]
        self.n_per_class = 600
        self.test_episodes = 600


class OmniglotCorruptTest(NumpyDataset):
    def __init__(self, root: str, split: str, n_way: int = 5, k_shot: int = 5, test_shots: int = 90, val_shots: int = 0, ood_test: bool = False) -> None:
        super().__init__(root, split, n_way, k_shot, test_shots, val_shots, ood_test)

        self.path = os.path.join(f"{n_way}-way", f"{k_shot}-shot", f"{test_shots}-testshot-{val_shots}-valshots")
        self.data = torch.from_numpy(np.load(os.path.join(root, "corruptions", "omniglot-resized-vinyals", "images_evaluation.npy")))

        self.n_classes = self.data.shape[0]
        self.n_corruptions = self.data.shape[1]
        self.levels = self.data.shape[2]

        self.dim = 28
        self.ch = 1
        self.n_per_class = 20
        self.test_episodes = 17000

    def getitem(self, i: int) -> Tuple[T, T, T, T]:
        spt_x, spt_y, qry_x, qry_y = self.get_empty()

        classes, idx = self.random_classes_and_shots(self.n_way, self.k_shot, self.test_shots)

        # hardcoded 15 here because we need to choose the number of shots according to the regular dataset, but it
        # will just be multiplied by a factor of 6 for 6 corruption intensities
        train_idx, test_idx = idx[:self.k_shot], idx[self.k_shot : self.k_shot + 15]
        corrs = np.random.choice(self.n_corruptions, size=classes.shape[0], replace=True)

        # get the regular test set
        for i, cl in enumerate(classes):
            c = corrs[i]

            tr, te = self.data[cl, c, 0, train_idx], self.data[cl, c, :, test_idx]
            tr = np.reshape(tr, (-1, 1, tr.shape[-2], tr.shape[-1]))
            te = np.reshape(te, (-1, 1, te.shape[-2], te.shape[-1]))
            if self.test_shots != 90:
                # this was a special test to see if we can even out the benefit of transductive batchnorm
                te = te[np.random.permutation(te.shape[0])[:self.test_shots]]

            spt_x[(i * self.k_shot) : ((i + 1) * self.k_shot)] = tr
            spt_y[(i * self.k_shot) : ((i + 1) * self.k_shot)] = i

            qry_x[(i * self.test_shots) : ((i + 1) * self.test_shots)] = te
            qry_y[(i * self.test_shots) : ((i + 1) * self.test_shots)] = i

        return 1 - (spt_x / 255.0), spt_y.long(), 1 - (qry_x / 255.0), qry_y.long()


class MiniImageNetCorruptTest(NumpyDataset):
    def __init__(self, root: str, split: str, n_way: int = 5, k_shot: int = 5, test_shots: int = 90, val_shots: int = 0, ood_test: bool = False) -> None:
        super().__init__(root, split, n_way, k_shot, test_shots, val_shots, ood_test)

        self.path = os.path.join(f"{n_way}-way", f"{k_shot}-shot", f"{test_shots}-testshot-{val_shots}-valshots")
        self.data = torch.from_numpy(np.load(os.path.join(root, "corruptions", "miniimagenet-c", "test.npy")))
        self.data = self.data.transpose(5, 3)
        # data is in the shape of (17, 6, 12000, 84, 84, 3) --> (corruptions, intensities, 20 clases * 600 instances, dim, dim, ch)
        # after transpose --> (17, 6, 12000, 3, 84, 84)

        self.n_classes = int(self.data.shape[2] / 600)
        self.n_corruptions = self.data.shape[0]
        self.levels = self.data.shape[2]

        self.dim = 84
        self.ch = 3
        self.n_per_class = 600
        self.test_episodes = 600 * 17

    def getitem(self, i: int) -> Tuple[T, T, T, T]:
        spt_x, spt_y, qry_x, qry_y = self.get_empty()

        classes, idx = self.random_classes_and_shots(self.n_way, self.k_shot, self.test_shots)

        # hardcoded 15 here because we need to choose the number of shots according to the regular dataset, but it
        # will just be multiplied by a factor of 6 for 6 corruption intensities
        train_idx, test_idx = idx[:self.k_shot], idx[self.k_shot : self.k_shot + 15]
        corrs = np.random.choice(self.n_corruptions, size=classes.shape[0], replace=True)

        # get the regular test set
        for i, cl in enumerate(classes):
            c = corrs[i]
            tr_idx, te_idx = cl * 600 + train_idx, cl * 600 + test_idx
            tr, te = self.data[c, 0, tr_idx], self.data[c, :, te_idx]
            # tr -> (shots, 3, 84, 84), te -> (6, test_shots, 3, 84, 84)

            te = np.reshape(te, (-1, 3, te.shape[-2], te.shape[-1]))
            spt_x[(i * self.k_shot) : ((i + 1) * self.k_shot)] = tr
            spt_y[(i * self.k_shot) : ((i + 1) * self.k_shot)] = i

            qry_x[(i * self.test_shots) : ((i + 1) * self.test_shots)] = te
            qry_y[(i * self.test_shots) : ((i + 1) * self.test_shots)] = i

        return (spt_x / 255.0), spt_y.long(), (qry_x / 255.0), qry_y.long()


ROOT = "st2"


def test_numpy_corr_loaders() -> None:
    for name, DS in zip(["omniglot", "miniimagenet"], [OmniglotCorruptTest, MiniImageNetCorruptTest]):
        for n_way, k_shot in zip([5, 5, 20, 20], [1, 5, 1, 5]):
            print(f"numpy {name} nway: {n_way} kshot: {k_shot}")
            dataset = DS(f"/{ROOT}/jeff/datasets", "test", n_way=n_way, k_shot=k_shot)
            spt_x, spt_y, qry_x, qry_y = dataset[0]
            print(f"support: {spt_x.size()} {spt_y.size()}")
            print(f"query: {qry_x.size()} {qry_y.size()}")

            ldr = DataLoader(dataset, batch_size=32, shuffle=True)
            for i, (spt_x, spt_y, qry_x, qry_y) in enumerate(ldr):
                print(spt_x.size(), spt_y.size(), qry_x.size(), qry_y.size())
                if i == 2:
                    break

            for i in range(10):
                spt_x, _, qry_x, _ = dataset[i]
                print(spt_x.size(), qry_x.size())
                path = f"data/examples/{name}-corrupt-numpy/plain-n-{n_way}-k-{k_shot}"
                os.makedirs(path, exist_ok=True)
                grid = torchvision.utils.make_grid(spt_x, nrow=k_shot)
                torchvision.utils.save_image(grid, os.path.join(path, f"support-{i}.png"))

                grid = torchvision.utils.make_grid(qry_x, nrow=15 * 6)
                torchvision.utils.save_image(grid, os.path.join(path, f"query-{i}.png"))


def test_numpy_loaders() -> None:
    for name, DS in zip(["omniglot", "miniimagenet"], [Omniglot, MiniImageNet]):
        for n_way, k_shot in zip([5, 5, 20, 20], [1, 5, 1, 5]):
            print(f"numpy {name} nway: {n_way} kshot: {k_shot}")
            dataset = DS(f"/{ROOT}/jeff/datasets", "train", n_way=n_way, k_shot=k_shot, test_shots=15, ood_test=False, _len=32 * 100)
            spt_x, spt_y, qry_x, qry_y = dataset[0]
            print(f"support: {spt_x.size()} {spt_y.size()}")
            print(f"query: {qry_x.size()} {qry_y.size()}")

            ldr = DataLoader(dataset, batch_size=32, shuffle=True)
            for i, (spt_x, spt_y, qry_x, qry_y) in enumerate(ldr):
                print(spt_x.size(), spt_y.size(), qry_x.size(), qry_y.size())
                if i == 2:
                    break

            for i in range(10):
                spt_x, _, qry_x, _ = dataset[i]
                print(spt_x.size(), qry_x.size())
                path = f"data/examples/{name}-numpy/plain-n-{n_way}-k-{k_shot}"
                os.makedirs(path, exist_ok=True)
                grid = torchvision.utils.make_grid(spt_x, nrow=k_shot)
                torchvision.utils.save_image(grid, os.path.join(path, f"support-{i}.png"))

                grid = torchvision.utils.make_grid(qry_x, nrow=15)
                torchvision.utils.save_image(grid, os.path.join(path, f"query-{i}.png"))

            del dataset

            print(f"numpy {name} WITH OOD CLASS TEST SET: nway: {n_way} kshot: {k_shot}")
            dataset = DS(f"/{ROOT}/jeff/datasets", "train", ood_test=True, n_way=n_way, k_shot=k_shot, _len=32 * 100)
            spt_x, spt_y, qry_x, qry_y = dataset[0]
            print(f"support: {spt_x.size()} {spt_y.size()}")
            print(f"query: {qry_x.size()} {qry_y.size()}")

            ldr = DataLoader(dataset, batch_size=32, shuffle=True)
            for i, (spt_x, spt_y, qry_x, qry_y) in enumerate(ldr):
                print(spt_x.size(), spt_y.size(), qry_x.size(), qry_y.size())
                if i == 2:
                    break

            for i in range(10):
                spt_x, _, qry_x, _ = dataset[i]
                print(spt_x.size(), qry_x.size())
                path = f"data/examples/{name}-numpy/plain-ood-class-metatest-n-{n_way}-k-{k_shot}"
                os.makedirs(path, exist_ok=True)
                grid = torchvision.utils.make_grid(spt_x, nrow=k_shot)
                torchvision.utils.save_image(grid, os.path.join(path, f"support-{i}.png"))

                grid = torchvision.utils.make_grid(qry_x, nrow=15)
                torchvision.utils.save_image(grid, os.path.join(path, f"query-{i}.png"))


if __name__ == "__main__":
    test_numpy_corr_loaders()
    test_numpy_loaders()
