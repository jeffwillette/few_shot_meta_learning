import abc
import os
import random
import sys
import typing

import higher
import numpy as np
import torch
import torchvision
from higher import patch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder
from torch.utils.tensorboard import SummaryWriter

from _utils import IdentityNet, get_episodes, train_val_split
from cnn import CNN2
from CommonModels import CNN, MiniCNN, ResNet18
from EpisodeGenerator import ImageFolderGenerator, OmniglotLoader
from utils import ece_yhat_only

# --------------------------------------------------
# Default configuration
# --------------------------------------------------
config = {} # initialize a configuration dictionary

# Hardware
config['device'] = torch.device('cuda:0' if torch.cuda.is_available() \
    else torch.device('cpu'))

# Dataset
config['datasource'] = 'omniglot-py'
config['suffix'] = 'png'  # extension of image file: png, jpg
config['image_size'] = (1, 64, 64)
config['ds_folder'] = './datasets'  # path to the folder containing the dataset
config['load_images'] = True # load images on RAM for fast access. Set False for large dataset to avoid out-of-memory

# Meta-learning method
config['ml_algorithm'] = 'maml'  # either: maml and vampire
config['first_order'] = True # applicable for MAML-like algorithms
config['num_models'] = 1 # number of models used in Monte Carlo to approximate expectation
config['KL_weight'] = 1e-4

# Task-related
config['max_way'] = 5
config['n_way'] = 5
config['k_shot'] = 1
config['v_shot'] = 15

# Training related parameters
config['network_architecture'] = 'CNN' # either CNN or ResNet18 specified in the CommonModels.py
config['batchnorm'] = False
config['num_inner_updates'] = 5
config['inner_lr'] = 0.1
config['meta_lr'] = 1e-3
config['minibatch'] = 20 # mini-batch of tasks
config['minibatch_print'] = np.lcm(config['minibatch'], 500)
config['num_episodes_per_epoch'] = 10000 # save model after every xx tasks
config['num_epochs'] = 1
config['resume_epoch'] = 0
# config['train_flag'] = True

# Testing
config['num_episodes'] = 100
config['episode_file'] = None  # path to a csv file with row as episode name and column as list of classes that form an episode

# Log
config['logdir'] = os.path.join('/media/n10/Data', 'meta_learning', config['ml_algorithm'], config['datasource'], config['network_architecture'])

# --------------------------------------------------
# Meta-learning class
# --------------------------------------------------
class MLBaseClass(object):
    """Meta-learning class for MAML and VAMPIRE
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, config: dict = config) -> None:
        """Initialize an instance of a meta-learning algorithm
        """
        self.config = config
        return
    
    @abc.abstractmethod
    def load_model(self, resume_epoch: int = None, **kwargs) -> typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer]:
        """Load the model for meta-learning algorithm
        """
        return

    @abc.abstractmethod
    def adapt_and_predict(
        self,
        model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer],
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        x_v: torch.Tensor,
        y_v: torch.Tensor
    ) -> typing.Tuple[higher.patch._MonkeyPatchBase, typing.List[torch.Tensor]]:
        """Adapt to task and predict labels of queried data
        """
        return
    
    @abc.abstractmethod
    def loss_extra(self, **kwargs) -> typing.Union[torch.Tensor, float]:
        """Calculate the loss on training subset using the task-specific parameter and regularization due to the prior of the meta-parameter. This is applicable for ABML and VAMPIRE.
        """
        return
    
    @abc.abstractmethod
    def KL_divergence(self, **kwargs) -> typing.Union[torch.Tensor, float]:
        """KL divergence between the task-specific q(w; lambda_i) and meta p(w; theta)
        """
        return

    @abc.abstractmethod
    def loss_prior(self, model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer], **kwargs) -> typing.Union[torch.Tensor, float]:
        """Loss prior or regularization for the meta-parameter
        """
        return

    def train(self, eps_generator: typing.Union[OmniglotLoader, ImageFolderGenerator]) -> None:
        """Train meta-learning model

        Args:
            eps_generator: the generator that generate episodes/tasks
        """
        print('Training is started.\nLog is stored at {0:s}.\n'.format(self.config['logdir']))

        # initialize/load model. Please see the load_model method implemented in each specific class for further information about the model
        model = self.load_model(resume_epoch=self.config['resume_epoch'], hyper_net_class=self.hyper_net_class, eps_generator=eps_generator)
        model[-1].zero_grad()

        # get list of episode names, each episode name consists of classes
        eps = get_episodes(episode_file_path=self.config['episode_file'])

        # initialize a tensorboard summary writer for logging
        tb_writer = SummaryWriter(
            log_dir=self.config['logdir'],
            purge_step=self.config['resume_epoch'] * self.config['num_episodes_per_epoch'] // self.config['minibatch_print'] if self.config['resume_epoch'] > 0 else None
        )

        try:
            for epoch_id in range(self.config['resume_epoch'], self.config['resume_epoch'] + self.config['num_epochs'], 1):
                print(f"Starting epoch: {epoch_id}")
                loss_monitor = 0.
                KL_monitor = 0.
                correct, total = 0., 0.
                for eps_count in range(self.config['num_episodes_per_epoch']):
                    # -------------------------
                    # get eps from the given csv file or just random (None)
                    # -------------------------
                    eps_name = random.sample(population=eps, k=1)[0]

                    # -------------------------
                    # episode data
                    # -------------------------
                    x_t, y_t, x_v, y_v = eps_generator.generate_episode(episode_name=eps_name)
                    x_t, y_t, x_v, y_v = x_t.cuda(), y_t.cuda(), x_v.cuda(), y_v.cuda()

                    # -------------------------
                    # adapt and predict the support data
                    # -------------------------

                    f_hyper_net, logits = self.adapt_and_predict(model=model, x_t=x_t, y_t=y_t, x_v=x_v, y_v=y_v)

                    loss_v = 0.
                    for logits_ in logits:
                        loss_v_temp = torch.nn.functional.cross_entropy(input=logits_, target=y_v)
                        loss_v = loss_v + loss_v_temp
                    loss_v = loss_v / len(logits)
                    loss_monitor += loss_v.item()  # monitor validation loss

                    correct += (torch.stack(logits).mean(dim=0).argmax(dim=-1) == y_v).sum().item()
                    total += y_v.numel()

                    # calculate KL divergence
                    KL_div = self.KL_divergence(model=model, f_hyper_net=f_hyper_net)
                    KL_monitor += KL_div.item() if isinstance(KL_div, torch.Tensor) else KL_div  # monitor KL divergence

                    # extra loss applicable for ABML only
                    loss_extra = self.loss_extra(model=model, f_hyper_net=f_hyper_net, x_t=x_t, y_t=y_t)

                    # accumulate KL divergence to loss
                    loss_v = loss_v + loss_extra + self.config['KL_weight'] * KL_div
                    loss_v = loss_v / self.config['minibatch']

                    # calculate gradients w.r.t. hyper_net's parameters
                    loss_v.backward()

                    # sys.stdout.write('\033[F')
                    # print(f"correct: {(correct / total):.4f}")  # , " ll: ", -np.log(ll / (i + 1)))

                    self.config['iters'] += 1

                    # update meta-parameters
                    if ((eps_count + 1) % self.config['minibatch'] == 0):
                        loss_prior = self.loss_prior(model=model)
                        if hasattr(loss_prior, 'requires_grad'):
                            loss_prior.backward()

                        model[-1].step()
                        model[-1].zero_grad()

                        # monitoring
                        if (eps_count + 1) % self.config['minibatch_print'] == 0:
                            loss_monitor /= self.config['minibatch_print']
                            KL_monitor = KL_monitor * self.config['minibatch'] / self.config['minibatch_print']

                            # calculate step for Tensorboard Summary Writer
                            global_step = (epoch_id * self.config['num_episodes_per_epoch'] + eps_count + 1) // self.config['minibatch_print']

                            tb_writer.add_scalar(tag='Cls loss', scalar_value=loss_monitor, global_step=global_step)
                            tb_writer.add_scalar(tag='KL divergence', scalar_value=KL_monitor, global_step=global_step)

                            # reset monitoring variables
                            loss_monitor = 0.
                            KL_monitor = 0.

                print(f"epoch {epoch_id} acc: {correct / total}")
                # save model
                checkpoint = {
                    'hyper_net_state_dict': model[0].state_dict(),
                    'opt_state_dict': model[-1].state_dict(),
                    'epoch': epoch_id,
                    'iters': self.config['iters']
                }

                checkpoint_path = os.path.join(self.config['logdir'], 'run_{}.pt'.format(self.config['run']))
                torch.save(obj=checkpoint, f=checkpoint_path)
                print('State dictionaries are saved into {0:s}\n'.format(checkpoint_path))

            print('Training is completed.')
        finally:
            print('\nClose tensorboard summary writer')
            tb_writer.close()

        return None

    def evaluate(self, eps_generator: typing.Union[OmniglotLoader, ImageFolderGenerator]) -> None:
        """Evaluate the performance
        """
        print('Evaluation is started.\n')
        # load model
        model = self.load_model(resume_epoch=self.config['resume_epoch'], hyper_net_class=self.hyper_net_class, eps_generator=eps_generator)

        # get list of episode names, each episode name consists of classes
        eps = get_episodes(episode_file_path=self.config['episode_file'], num_episodes=1000)

        ece, roc_auc, ll = 0., 0., 0.
        correct, total = 0., 0.
        for i, eps_name in enumerate(eps):
            x_t, y_t, x_v, y_v = eps_generator.generate_episode(episode_name=eps_name)
            x_t, y_t, x_v, y_v = x_t.cuda(), y_t.cuda(), x_v.cuda(), y_v.cuda()
            # split data into train and validation

            # grid = torchvision.utils.make_grid(x_t, nrow=5)
            # torchvision.utils.save_image(grid, f"train-{i}.png")

            # grid = torchvision.utils.make_grid(x_v, nrow=90)
            # torchvision.utils.save_image(grid, f"test-{i}.png")

            # if i == 5:
            #     exit()

            # print(x_v.size())
            _, logits = self.adapt_and_predict(model=model, x_t=x_t, y_t=y_t, x_v=x_v, y_v=None)

            # initialize y_prediction
            y_pred = torch.zeros((y_v.shape[0], self.config["n_way"]), device=x_v.device)
            for logits_ in logits:
                y_pred += torch.softmax(input=logits_, dim=1)
            y_pred /= len(logits)
            # exit()

            correct += (y_pred.argmax(dim=1) == y_v).sum()
            total += y_v.numel()

            e, _, _ = ece_yhat_only(100, y_v, y_pred, device=y_v.device)
            ece += e

            yhot = OneHotEncoder(categories='auto').fit_transform(y_v.cpu().numpy().reshape(-1, 1)).toarray()
            roc_auc += roc_auc_score(yhot, y_pred.detach().cpu().numpy(), multi_class="ovr")

            ll += y_pred[torch.arange(y_v.size(0)), y_v].detach().cpu().sum()

            sys.stdout.write('\033[F')
            print(i + 1)  # , " ll: ", -np.log(ll / (i + 1)))

        acc = correct / total
        ece = ece / (i + 1)
        roc_auc = roc_auc / (i + 1)
        nll = -np.log(ll) + np.log(total)

        files = ["{}_acc_{}.txt", "{}_ece_{}.txt", "{}_auroc_{}.txt", "{}_nll_{}.txt"]
        metrics = [acc, ece, roc_auc, nll]
        for fl, mtrc in zip(files, metrics):
            if self.config["corrupt"]:
                extra = "corrupt"
            elif self.config["ood_test"]:
                extra = "oodtest"
            else:
                extra = "plain"

            path = os.path.join(self.config['logdir'], fl.format(extra, self.config['run']))
            with open(path, "a+") as f:
                f.write("{}\n".format(mtrc))

        print('\nAccuracy: {0:.2f}\nECE: {1:.2f}\nAUROC: {2:.2f}\nNLL: {3:.2f}'.format(acc * 100, ece, roc_auc, nll))

    # --------------------------------------------------
    # Auxilliary functions for MAML-like algorithms
    # --------------------------------------------------
    @staticmethod
    def torch_module_to_functional(torch_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
        """Convert a conventional torch module to its "functional" form
        """
        f_net = higher.patch.make_functional(module=torch_net)
        f_net.track_higher_grads = False
        f_net._fast_params = [[]]

        return f_net

    def adapt_to_episode(self, x: torch.Tensor, y: torch.Tensor, hyper_net: torch.nn.Module, f_base_net: higher.patch._MonkeyPatchBase, train_flag: bool = True) -> higher.patch._MonkeyPatchBase:
        """Inner-loop for MAML-like algorithm

        Args:
            x, y: training data and corresponding labels
            hyper_net: the meta-model
            f_base_net: the functional form of the based neural network
            kl_div_fn: function that calculates the KL divergence

        Returns: the task-specific meta-model
        """
        # convert hyper_net to its functional form
        f_hyper_net = higher.patch.monkeypatch(
            module=hyper_net,
            copy_initial_weights=False,
            track_higher_grads=train_flag
        )

        hyper_net_params = [p for p in hyper_net.parameters()]

        for _ in range(self.config['num_inner_updates']):
            grads_accum = [0] * len(hyper_net_params)  # accumulate gradients of Monte Carlo sampling

            q_params = f_hyper_net.fast_params  # parameters of the task-specific hyper_net

            # KL divergence
            KL_div = self.KL_divergence(p=hyper_net_params, q=q_params)

            for i in range(self.config['num_models']):
                base_net_params = f_hyper_net.forward()

                y_logits = f_base_net.forward(x, params=base_net_params)

                cls_loss = torch.nn.functional.cross_entropy(input=y_logits, target=y)

                loss = cls_loss + self.config['KL_weight'] * KL_div

                if self.config['first_order']:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=q_params,
                        retain_graph=True
                    )
                    # print(f"grads: {grads}")
                else:
                    grads = torch.autograd.grad(
                        outputs=loss,
                        inputs=q_params,
                        create_graph=True
                    )

                # accumulate gradients
                for i in range(len(grads)):
                    grads_accum[i] = grads_accum[i] + grads[i] / self.config['num_models']

            new_q_params = []
            for param, grad in zip(q_params, grads_accum):
                new_q_params.append(higher.optim._add(tensor=param, a1=-self.config['inner_lr'], a2=grad))

            f_hyper_net.update_params(new_q_params)

        return f_hyper_net

    def predict(self, x: torch.Tensor, f_hyper_net: higher.patch._MonkeyPatchBase, f_base_net: higher.patch._MonkeyPatchBase) -> typing.List[torch.Tensor]:
        """Make prediction

        Args:
            x: input data
            f_hyper_net: task-specific meta-model
            f_base_net: functional form of the base neural network

        Returns: a list of logits predicted by the task-specific meta-model
        """
        logits = [None] * self.config['num_models']
        for model_id in range(self.config['num_models']):
            base_net_params = f_hyper_net.forward()
            logits_temp = f_base_net.forward(x, params=base_net_params)

            logits[model_id] = logits_temp

        return logits

    def load_maml_like_model(self, resume_epoch: int = None, **kwargs) -> typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer]:
        """Initialize or load the hyper-net and base-net models

        Args:
            hyper_net_class: point to the hyper-net class of interest: IdentityNet for MAML or NormalVariationalNet for VAMPIRE
            resume_epoch: the index of the file containing the saved model

        Returns: a tuple consisting of
            hypet_net: the hyper neural network
            base_net: the base neural network
            meta_opt: the optimizer for meta-parameter
        """
        if resume_epoch is None:
            resume_epoch = self.config['resume_epoch']

        if self.config['network_architecture'] == 'CNN':
            base_net = CNN(
                dim_output=self.config['n_way'],
                bn_affine=self.config['batchnorm']
            )
        elif self.config['network_architecture'] == 'CNN2':
            in_dim = 1 if "omniglot" in self.config["datasource"] else 3
            base_net = CNN2(in_dim, out_dim=self.config["n_way"])
        elif self.config['network_architecture'] == 'ResNet18':
            base_net = ResNet18(
                dim_output=self.config['n_way'],
                bn_affine=self.config['batchnorm']
            )
        elif self.config['network_architecture'] == 'MiniCNN':
            base_net = MiniCNN(dim_output=self.config['n_way'], bn_affine=self.config['batchnorm'])
        else:
            raise NotImplementedError('Network architecture is unknown. Please implement it in the CommonModels.py.')

        # ---------------------------------------------------------------
        # run a dummy task to initialize lazy modules defined in base_net
        # ---------------------------------------------------------------
        x_t, _, _, _ = kwargs['eps_generator'].generate_episode(episode_name=None)

        # run to initialize lazy modules
        base_net(x_t)
        params = torch.nn.utils.parameters_to_vector(parameters=base_net.parameters())
        print('Number of parameters of the base network = {0:d}.\n'.format(params.numel()))

        hyper_net = kwargs['hyper_net_class'](base_net=base_net)

        # move to device
        base_net.to(self.config['device'])
        hyper_net.to(self.config['device'])

        # optimizer
        meta_opt = torch.optim.Adam(params=hyper_net.parameters(), lr=self.config['meta_lr'])

        # load model if there is saved file
        if resume_epoch > 0 or not self.config["train_flag"]:
            # path to the saved file
            checkpoint_path = os.path.join(self.config['logdir'], 'run_{}.pt'.format(self.config['run']))

            # load file
            saved_checkpoint = torch.load(
                f=checkpoint_path,
                map_location=lambda storage,
                loc: storage.cuda(self.config['device'].index) if self.config['device'].type == 'cuda' else storage
            )

            self.config['iters'] = saved_checkpoint['iters']
            self.config['resume_epoch'] = saved_checkpoint['epoch']
            print(f"Resuming from epoch: {self.config['resume_epoch']} iter: {self.config['iters']}")

            # load state dictionaries
            hyper_net.load_state_dict(state_dict=saved_checkpoint['hyper_net_state_dict'])
            meta_opt.load_state_dict(state_dict=saved_checkpoint['opt_state_dict'])

            # update learning rate
            for param_group in meta_opt.param_groups:
                if param_group['lr'] != self.config['meta_lr']:
                    param_group['lr'] = self.config['meta_lr']

        return hyper_net, base_net, meta_opt
