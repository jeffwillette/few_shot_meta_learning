from typing import List, Tuple, Any
import os

import numpy as np  # type: ignore
import random
import torch
from torch.utils.data import Dataset
from matplotlib.colors import to_rgba  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from matplotlib.lines import Line2D  # type: ignore
from sklearn.datasets import make_moons, make_circles  # type: ignore

T = torch.Tensor


def get_biased_sample_idx(x: Any, y: Any, k_shot: int) -> Tuple[Any, ...]:

    classes = np.unique(y)
    n_sections = 2  # (n-way + kshot) * classes needs to be equally divisible by n_sections

    sx, sy, qx, qy = np.empty((0, 2)), np.empty((0,)), np.empty((0, 2)), np.empty((0,))
    for c in classes:
        class_idx = np.argwhere(y == c).squeeze(1)
        class_x, class_y = x[class_idx], y[class_idx]

        x_or_y = 0 if np.sign(np.random.rand() - 0.5) < 0 else 1  # choose x or y index randomly
        section = np.random.permutation(n_sections)  # which half of the data to get
        x_idx = np.argsort(class_x[:, x_or_y])

        def sec(n: int) -> int:
            return int(n * (x_idx.shape[0] // n_sections))

        # get the support and qeury sets for this class which are split by section (whichever biased section we chose)
        spt_x = class_x[x_idx[sec(section[0]) : sec(section[0] + 1)]]  # get the proper third
        spt_y = class_y[x_idx[sec(section[0]) : sec(section[0] + 1)]]  # get the proper third
        qry_x = class_x[x_idx[sec(section[1]) : sec(section[1] + 1)]]
        qry_y = class_y[x_idx[sec(section[1]) : sec(section[1] + 1)]]

        # collect random k of the biased support sets into one and leave the rest for the qeury set
        spt_perm = np.random.permutation(spt_x.shape[0])
        sx = np.concatenate((sx, spt_x[spt_perm[:k_shot]]))
        sy = np.concatenate((sy, spt_y[spt_perm[:k_shot]]))
        qx = np.concatenate((qx, spt_x[spt_perm[k_shot:]], qry_x))
        qy = np.concatenate((qy, spt_y[spt_perm[k_shot:]], qry_y))

    return sx, sy, qx, qy


class ToyDataset(Dataset):
    def __init__(self, seed: int = 0, k_shot: int = 10, total_tasks: int = 100, test_shots: int = 50):

        self.seed = seed
        self.k_shot = k_shot
        self.total_tasks = total_tasks
        self.test_shots = test_shots

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self) -> int:
        return self.total_tasks


class MetaMoons(ToyDataset):
    def __init__(
        self,
        seed: int = 0,
        k_shot: int = 10,
        total_tasks: int = 100,
        test_shots: int = 50,
    ):
        super().__init__(seed=seed, k_shot=k_shot, total_tasks=total_tasks, test_shots=test_shots)

        self.n_way = 2
        self.name = "moons"
        self.path = os.path.join("toy-moons", "2-way", f"{k_shot}-shot", f"{test_shots}-testshot")

    def __getitem__(self, i: int) -> Tuple[T, T, T, T]:
        return self.gen_random_task()

    def sample_uniform(self) -> T:
        x = torch.linspace(-3, 3, 100)
        return torch.stack(torch.meshgrid(x, x), dim=-1).view(-1, 2)

    def gen_random_task(self) -> Tuple[T, T, T, T]:
        noise = np.random.rand() * .25
        x, y = make_moons(n_samples=self.n_way * (self.k_shot + self.test_shots), noise=noise, random_state=self.seed)

        sx, sy, qx, qy = get_biased_sample_idx(x, y, self.k_shot)
        sx, sy, qx, qy = torch.from_numpy(sx).float(), torch.from_numpy(sy).long(), torch.from_numpy(qx).float(), torch.from_numpy(qy).long()
        return sx, sy, qx, qy


class MetaCircles(ToyDataset):
    def __init__(
        self,
        seed: int = 0,
        k_shot: int = 10,
        total_tasks: int = 100,
        test_shots: int = 50,
    ):
        super().__init__(seed=seed, k_shot=k_shot, total_tasks=total_tasks, test_shots=test_shots)

        self.n_way = 2
        self.name = "circles"
        self.path = os.path.join("toy-circles", "2-way", f"{k_shot}-shot", f"{test_shots}-testshot")

    def __getitem__(self, i: int) -> Tuple[T, T, T, T]:
        return self.gen_random_task()

    def sample_uniform(self) -> T:
        x = torch.linspace(-3, 3, 100)
        return torch.stack(torch.meshgrid(x, x), dim=-1).view(-1, 2)

    def gen_random_task(self) -> Tuple[T, T, T, T]:
        noise = np.random.rand() * .25
        scale = np.random.rand() * 0.8
        x, y = make_circles(n_samples=self.k_shot + self.test_shots, noise=noise, factor=scale, random_state=self.seed)

        sx, sy, qx, qy = get_biased_sample_idx(x, y, self.k_shot)
        sx, sy, qx, qy = torch.from_numpy(sx).float(), torch.from_numpy(sy).long(), torch.from_numpy(qx).float(), torch.from_numpy(qy).long()
        return sx, sy, qx, qy


class RandomGaussians(ToyDataset):
    def __init__(
        self,
        seed: int = 0,
        n_way: int = 5,
        k_shot: int = 5,
        total_tasks: int = 100,
        test_shots: int = 15,
        mu_rng: List[int] = [-5, 5],
        var_rng: List[float] = [0.1, 1.0],
        dim: int = 2
    ):
        super().__init__(seed=seed, k_shot=k_shot, total_tasks=total_tasks, test_shots=test_shots)

        self.name = "2d-gaussians"
        self.mu_rng = mu_rng
        self.n_way = n_way
        self.var_rng = var_rng
        self.var = var_rng
        self.dim = dim
        self.name = "gausian"
        self.path = os.path.join("toy-gaussian", f"{n_way}-way", f"{k_shot}-shot", f"{test_shots}-testshot")

    def sample_uniform(self) -> T:
        x = torch.linspace(-3, 3, 100)
        return torch.stack(torch.meshgrid(x, x), dim=-1).view(-1, self.dim)

    def sample(self, N: torch.distributions.MultivariateNormal, variant: str = "uniform") -> Tuple[T, T]:
        train, test = N.sample((self.k_shot,)).transpose(0, 1), N.sample((self.test_shots,)).transpose(0, 1)
        return train, test

    def gen_random_task(self) -> Tuple[T, T, T, T]:
        # sample mus and sigmas uniformyl according to their range
        mus = torch.rand((self.n_way, self.dim)) * (self.mu_rng[1] - self.mu_rng[0]) + self.mu_rng[0]

        # decompose PSD sigma as O^TDO with orthogonal O's to make random PSD covariance
        # https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
        O = torch.rand((self.n_way, self.dim, self.dim)) * 2 - 1
        O = torch.qr(O)[0]
        D = torch.stack([torch.eye(self.dim) * torch.rand(self.dim) for i in range(self.n_way)])

        # make the eigenvectors be different lengths in order to make the direction elliptical ratio of 5:1
        tmp = (torch.rand((self.n_way, self.dim)) * (self.var_rng[1] - self.var_rng[0]) + self.var_rng[0]).unsqueeze(1)
        tmp[:, :, 1] = tmp[:, :, 0] / 5
        D = D * tmp
        sigmas = O.transpose(1, 2).bmm(D.bmm(O))

        N = torch.distributions.MultivariateNormal(mus, sigmas)
        labels = torch.randperm(self.n_way)

        train_x, test_x = self.sample(N)

        mu, sigma = train_x.mean(dim=(0, 1)), train_x.std(dim=(0, 1))

        train_x = (train_x - mu) / sigma
        test_x = (test_x - mu) / sigma
        train_y = labels.unsqueeze(-1).repeat(1, self.k_shot)
        test_y = labels.unsqueeze(-1).repeat(1, self.test_shots)

        train_x, train_y, test_x, test_y = train_x.reshape(-1, self.dim).numpy(), train_y.reshape(-1).numpy(), test_x.reshape(-1, self.dim).numpy(), test_y.reshape(-1).numpy()
        x, y = np.concatenate((train_x, test_x)), np.concatenate((train_y, test_y))

        assert x.shape[0] % 2 == 0, f"x needs to be evenly divisible by 2 (got shape {x.shape}) for the toy Gaussian, if not you have to fix 'get biased sample function'"
        sx, sy, qx, qy = get_biased_sample_idx(x, y, self.k_shot)

        return torch.from_numpy(sx).float(), torch.from_numpy(sy).long(), torch.from_numpy(qx).float(), torch.from_numpy(qy).long()

    def __getitem__(self, i: int) -> Tuple[T, T, T, T]:
        return self.gen_random_task()


colors = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    "mediumseagreen", "teal", "navy", "darkgoldenrod", "darkslateblue",
]


def get_color(i: int) -> Tuple[float, ...]:
    if i < len(colors):
        return to_rgba(colors[i])  # type: ignore
    return (np.random.rand(), np.random.rand(), np.random.rand(), 1.0)


BATCH_SIZE = 3
SEED = 1

if __name__ == "__main__":
    ds: Any
    do_plots = ["moons", "circles", "gaussian"]

    if "moons" in do_plots:
        ds = MetaMoons(seed=SEED)
        fig, axes = plt.subplots(nrows=1, ncols=BATCH_SIZE, figsize=(BATCH_SIZE * 7, 6))
        for i, ax in enumerate(axes):
            xtr, ytr, xte, yte = ds[0]

            # this sample will be form a different task, but we are only taking the uniform noise so it is ok

            ax.scatter(xtr[:, 0], xtr[:, 1], c=[get_color(v.item()) for v in ytr], s=50, edgecolors=(0, 0, 0, 0.5), linewidths=2.0)
            ax.scatter(xte[:, 0], xte[:, 1], c=[get_color(v.item()) for v in yte], marker='*', s=20)
            ax.set_title(f"task: {i}")
            if i == BATCH_SIZE - 1:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label='train', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='*', color='w', label='test', markerfacecolor='black', markersize=10),
                ]
                ax.legend(handles=legend_elements)

        path = os.path.join("data", "examples", "toy-moons")
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, "metatrain-example.pdf"))
        fig.savefig(os.path.join(path, "metatrain-example.png"))

    if "circles" in do_plots:
        ds = MetaCircles(seed=SEED)
        fig, axes = plt.subplots(nrows=1, ncols=BATCH_SIZE, figsize=(BATCH_SIZE * 7, 6))
        for i, ax in enumerate(axes):
            xtr, ytr, xte, yte = ds[0]

            # this sample will be form a different task, but we are only taking the uniform noise so it is ok

            ax.scatter(xtr[:, 0], xtr[:, 1], c=[get_color(v.item()) for v in ytr], s=50, edgecolors=(0, 0, 0, 0.5), linewidths=2.0)
            ax.scatter(xte[:, 0], xte[:, 1], c=[get_color(v.item()) for v in yte], marker='*', s=20)
            ax.set_title(f"task: {i}")
            if i == BATCH_SIZE - 1:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label='train', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='*', color='w', label='test', markerfacecolor='black', markersize=10),
                ]
                ax.legend(handles=legend_elements)

        path = os.path.join("data", "examples", "toy-circles")
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, "metatrain-example.pdf"))
        fig.savefig(os.path.join(path, "metatrain-example.png"))

    if "gaussian" in do_plots:
        # RANDOM GAUSSIANS
        ds = RandomGaussians(seed=SEED, k_shot=5, test_shots=15)
        fig, axes = plt.subplots(nrows=1, ncols=BATCH_SIZE, figsize=(BATCH_SIZE * 7, 6))
        for i, ax in enumerate(axes):
            xtr, ytr, xte, yte = ds[0]
            ax.scatter(xtr[:, 0], xtr[:, 1], c=[get_color(v.item()) for v in ytr], s=50, edgecolors=(0, 0, 0, 0.5), linewidths=2.0)
            ax.scatter(xte[:, 0], xte[:, 1], c=[get_color(v.item()) for v in yte], marker='*', s=20)
            ax.set_title(f"task: {i}")
            if i == BATCH_SIZE - 1:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label='train', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='*', color='w', label='test', markerfacecolor='black', markersize=10),
                ]
                ax.legend(handles=legend_elements)

        path = os.path.join("data", "examples", "toy-gaussian")
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, "metatrain-example.pdf"))
        fig.savefig(os.path.join(path, "metatrain-example.png"))
