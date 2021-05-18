import os
import typing

import higher
import torch

from _utils import NormalVariationalNet, kl_divergence_gaussians
from CommonModels import CNN, ResNet18
from MLBaseClass import MLBaseClass


class Abml(MLBaseClass):
    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

        self.hyper_net_class = NormalVariationalNet

        # prior parameters
        self.gamma_prior = torch.distributions.gamma.Gamma(concentration=1, rate=0.01)
        self.normal_prior = torch.distributions.normal.Normal(loc=0, scale=1)

    def load_model(self, resume_epoch: int = None, **kwargs) -> typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer]:
        """Initialize or load the hyper-net and base-net models

        Args:
            hyper_net_class: point to the hyper-net class of interest: IdentityNet for MAML or NormalVariationalNet for VAMPIRE
            resume_epoch: the index of the file containing the saved model

        Returns: a tuple consisting of
            hypet_net: the hyper neural network
            base_net: the base neural network
            meta_opt: the optimizer for meta-parameter
        """
        return self.load_maml_like_model(resume_epoch=resume_epoch, **kwargs)

    def adapt_and_predict(self, model: typing.Tuple[torch.nn.Module, typing.Optional[higher.patch._MonkeyPatchBase], torch.optim.Optimizer], x_t: torch.Tensor, y_t: torch.Tensor, x_v: torch.Tensor, y_v: torch.Tensor) -> typing.Tuple[higher.patch._MonkeyPatchBase, typing.List[torch.Tensor]]:
        """Adapt and predict the labels of the queried data
        """
        # -------------------------
        # adapt on the support data
        # -------------------------
        f_base_net = self.torch_module_to_functional(torch_net=model[1])
        f_hyper_net = self.adapt_to_episode(x=x_t, y=y_t, hyper_net=model[0], f_base_net=f_base_net, train_flag=True)

        # -------------------------
        # predict labels of queried data
        # -------------------------
        logits = self.predict(x=x_v, f_hyper_net=f_hyper_net, f_base_net=f_base_net)

        return f_hyper_net, logits

    def loss_extra(self, **kwargs) -> typing.Union[torch.Tensor, float]:
        """Loss on the training subset
        """
        f_base_net = self.torch_module_to_functional(torch_net=kwargs['model'][1])
        logits = self.predict(x=kwargs['x_t'], f_hyper_net=kwargs['f_hyper_net'], f_base_net=f_base_net)
        loss = 0.
        for logits_ in logits:
            loss_temp = torch.nn.functional.cross_entropy(input=logits_, target=kwargs['y_t'])
            loss = loss + loss_temp

        loss = loss / len(logits)

        return loss

    def loss_prior(self, model, **kwargs) -> typing.Union[torch.Tensor, float]:
        """Loss prior or regularization for the meta-parameter
        """
        regularization = 0.

        hyper_net_params = [p for p in model[0].parameters()]
        for i, param in enumerate(hyper_net_params):
            if i < (len(hyper_net_params) // 2):
                regularization = regularization - self.normal_prior.log_prob(value=param).sum()
            else:
                tau = torch.exp(-2 * param)
                regularization = regularization - self.gamma_prior.log_prob(value=tau).sum()

        # regularization is weighted by inverse of the number of mini-batches used.
        # However, the number of mini-batches might change since one might want to train less or more.
        # For simplicity, the KL_weight is used as the weighting factor.
        regularization = regularization * self.config['KL_weight']

        return regularization

    @staticmethod
    def KL_divergence(**kwargs) -> typing.Union[torch.Tensor, float]:
        """
        """
        if ('p' not in kwargs) or ('q' not in kwargs):
            p = [p_ for p_ in kwargs['model'][0].parameters()]
            q = kwargs['f_hyper_net'].fast_params
        else:
            p = kwargs['p']
            q = kwargs['q']

        KL_div = kl_divergence_gaussians(p=p, q=q)

        return KL_div
