import os

import torch


def main() -> None:
    p = os.path.join(".logs", "meta_learning")
    for model in os.listdir(p):
        for ds in os.listdir(os.path.join(p, model)):
            for arch in os.listdir(os.path.join(p, model, ds)):
                for exp in os.listdir(os.path.join(p, model, ds, arch)):
                    for fl in os.listdir(os.path.join(p, model, ds, arch, exp)):
                        if ".pt" in fl:
                            sd = torch.load(os.path.join(p, model, ds, arch, exp, fl))
                            print(f"path: {os.path.join(p, model, ds, arch, exp)} epoch: {sd['epoch']} iters: {sd['iters']}")


if __name__ == "__main__":
    main()
