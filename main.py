"""
# MAML
python3 main.py --datasource=omniglot-py --suffix=png --load-images --ml-algorithm=MAML --num-models=1 --first-order --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=5 --num-epochs=100 --resume-epoch=0 --train

python3 main.py --datasource=miniImageNet --suffix=jpg --load-images --ml-algorithm=MAML --first-order --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=5 --num-epochs=100 --resume-epoch=0 --train

# VAMPIRE
python3 main.py --datasource=omniglot-py --suffix=png --load-images --ml-algorithm=vampire --num-models=2 --first-order --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=5 --num-epochs=100 --resume-epoch=0 --train

python3 main.py --datasource=miniImageNet --suffix=jpg --load-images --ml-algorithm=vampire --num-models=2 --first-order --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=5 --num-epochs=100 --resume-epoch=0 --train

# ABML
python3 main.py --datasource=omniglot-py --suffix=png --load-images --ml-algorithm=abml --num-models=2 --first-order --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=5 --num-epochs=100 --resume-epoch=0 --train

python3 main.py --datasource=miniImageNet --suffix=jpg --load-images --ml-algorithm=abml --num-models=2 --first-order --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=5 --num-epochs=100 --resume-epoch=0 --train

# PROTONET
python3 main.py --datasource=omniglot-py --suffix=png --load-images --ml-algorithm=protonet --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=10 --num-epochs=100 --resume-epoch=0 --train

python3 main.py --datasource=miniImageNet --suffix=jpg --load-images --ml-algorithm=protonet --network-architecture=CNN --no-batchnorm --min-way=5 --max-way=10 --num-epochs=100 --resume-epoch=0 --train
"""
import torch
import numpy as np
import os
import argparse

# from MetaLearning import MetaLearning
from Maml import Maml
from typing import Any
from Vampire import Vampire
from Abml import Abml
from ProtoNet import ProtoNet
from data import Omniglot, MiniImageNet, OmniglotCorruptTest, MiniImageNetCorruptTest
# --------------------------------------------------
# SETUP INPUT PARSER
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--datasource', type=str, default='omniglot-py', help='Dataset: omniglot, ImageNet')

parser.add_argument('--load-images', dest='load_images', action='store_true')
parser.add_argument('--no-load-images', dest='load_images', action='store_false')
parser.set_defaults(load_images=True)

parser.add_argument('--ml-algorithm', type=str, default='MAML', help='Few-shot learning methods, including: MAML, vampire or protonet')

parser.add_argument('--first-order', dest='first_order', action='store_true')
parser.add_argument('--no-first-order', dest='first_order', action='store_false')
parser.set_defaults(first_order=True)
parser.add_argument('--KL-weight', type=float, default=1e-6, help='Weighting factor for the KL divergence (only applicable for VAMPIRE)')

parser.add_argument('--network-architecture', type=str, default='ResNet12', help='The base model used, including CNN and ResNet18 defined in CommonModels')

# Including learnable BatchNorm in the model or not learnable BN
parser.add_argument('--batchnorm', dest='batchnorm', action='store_true')
parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false')
parser.set_defaults(batchnorm=False)

parser.add_argument('--n-way', type=int, default=5, help='Maximum number of classes within an episode')

parser.add_argument('--num-inner-updates', type=int, default=5, help='The number of gradient updates for episode adaptation')
parser.add_argument('--inner-lr', type=float, default=0.1, help='Learning rate of episode adaptation step')

parser.add_argument('--ds-folder', type=str, default='../datasets', help='Parent folder containing the dataset')
parser.add_argument('--logdir', type=str, default='.logs', help='Folder to store model and logs')

parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate for meta-update')
parser.add_argument('--minibatch', type=int, default=20, help='Minibatch of episodes to update meta-parameters')

parser.add_argument('--k-shot', type=int, default=1, help='Number of training examples per class')
parser.add_argument('--v-shot', type=int, default=15, help='Number of validation examples per class')

parser.add_argument('--num-episodes-per-epoch', type=int, default=10000, help='Save meta-parameters after this number of episodes')
parser.add_argument('--num-epochs', type=int, default=1, help='')
parser.add_argument('--resume-epoch', type=int, default=0, help='Resume')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--num-models', type=int, default=1, help='Number of base network sampled from the hyper-net')

parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes used in testing')
parser.add_argument('--episode-file', type=str, default=None, help='Path to csv file: row = episode, columns = list of classes within the episode')

parser.add_argument('--run', type=int, default=0, help='the run number to append to the paths')
parser.add_argument('--corrupt', action="store_true", help='whetehr or not to run the corrupted test set')

args = parser.parse_args()
print()

config = {}
for key in args.__dict__:
    config[key] = args.__dict__[key]

subdir = f"{config['n_way']}-{config['k_shot']}"
config['logdir'] = os.path.join(config['logdir'], 'meta_learning', config['ml_algorithm'].lower(), config['datasource'], config['network_architecture'], subdir)
if not os.path.exists(path=config['logdir']):
    from pathlib import Path
    Path(config['logdir']).mkdir(parents=True, exist_ok=True)

config['minibatch_print'] = np.lcm(config['minibatch'], 500)

config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))
config['iters'] = 0

if __name__ == "__main__":
    # task/episode generator

    print(config['datasource'], args.corrupt)
    EpisodeGeneratorClass: Any
    if config['datasource'] in ['omniglot'] and not args.corrupt:
        EpisodeGeneratorClass = Omniglot
    elif config['datasource'] in ['miniimagenet'] and not args.corrupt:
        EpisodeGeneratorClass = MiniImageNet
    elif config['datasource'] in ['omniglot'] and config['corrupt']:
        EpisodeGeneratorClass = OmniglotCorruptTest
    elif config['datasource'] in ['miniimagenet'] and config['corrupt']:
        EpisodeGeneratorClass = MiniImageNetCorruptTest
    else:
        raise ValueError('Unknown dataset')

    eps_generator = EpisodeGeneratorClass(
        root=os.path.join(config['ds_folder']),
        split="train" if config["train_flag"] else "test",
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        test_shots=config['v_shot'],
        # ood_test=True,
    )

    ml_algorithms = {
        'Maml': Maml,
        'Vampire': Vampire,
        'Abml': Abml,
        'Protonet': ProtoNet
    }
    print('ML algorithm = {0:s}'.format(config['ml_algorithm']))

    # Initialize a meta-learning instance
    ml = ml_algorithms[config['ml_algorithm'].capitalize()](config=config)

    if config['train_flag']:
        ml.train(eps_generator=eps_generator)
    else:
        ml.evaluate(eps_generator=eps_generator)
