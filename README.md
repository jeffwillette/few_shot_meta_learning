# Few-shot meta-learning
This repository contains the implementations of many meta-learning algorithms to solve the few-shot learning problem in PyTorch, including:
- [Model-Agnostic Meta-Learning (MAML)](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf)
- ~~[Probabilistic Model-Agnostic Meta-Learning (PLATIPUS)](https://papers.nips.cc/paper/8161-probabilistic-model-agnostic-meta-learning.pdf)~~
- [Prototypical Networks (protonet)](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)
- [Bayesian Model-Agnostic Meta-Learning (BMAML)](https://papers.nips.cc/paper/7963-bayesian-model-agnostic-meta-learning.pdf)
- [Amortized Bayesian Meta-Learning](https://openreview.net/pdf?id=rkgpy3C5tX)
- [Uncertainty in Model-Agnostic Meta-Learning using Variational Inference (VAMPIRE)](http://openaccess.thecvf.com/content_WACV_2020/papers/Nguyen_Uncertainty_in_Model-Agnostic_Meta-Learning_using_Variational_Inference_WACV_2020_paper.pdf)

## Python package requirements
- PyTorch __1.8__ (which intrioduces the new Lazy module)
- [higher][higher repo]

## New updates with functional form of torch module
What does "functional" mean? It is similar to the module `torch.nn.functional`, where the parameters can be handled explicitly, not implicitly as in PyTorch `torch.nn.Sequential()`. For example, one can construct a 2-hidden-layer fully-connected neural network from PyTorch under `fc_model`:
```python
# conventional with implicitly-handled parameter
y = fc_model(x) # parameters are handled by PyTorch implicitly

# functional form
y = fc_model(x, theta) # theta is the parameter and fc_model is the functional.
```

With the current PyTorch, one needs to manually implement the "functional" form of every component of the model of interest via `torch.nn.functional`. This is, however, inconvenient when changing the network architecture.

Fortunately, Facebook Research has developed [__higher__][higher repo] - a library that can easily convert any "conventional" neural network into its "functional" form. For example:
```python
# define a network
resnet18 = torchvision.models.resnet18(pretrain=False)

# get its parameters
params = list(resnet18.paramters())

# convert the network to its functional form
f_resnet18 = higher.patch.make_functional(restnet18)

# forward with functional and handling parameter explicitly
y1 = f_resnet18.forward(x=x1, params=params)

# update parameter
new_params = update_parameter(params)

# forward on different data with new paramter
y2 = f_resnet18.forward(x=x2, params=new_params)
```

Hence, we only need to load or specify the "conventional" model written in PyTorch without manually re-implementing its "functional" form. I have made a few common models in `CommonModels.py`, so that we can load them to use immediately.

Although [__higher__][higher repo] provides convenient APIs to track gradients, it does not allow us to use the "first-order" approximate, resulting in more memory and longer training time. I have created a work-around solution to enable the "first-order" approximation, and controlled this by setting `--first-order=True` when running the code.

Current update is available for most popular algorithms, including MAML, ABML, VAMPIRE and ProtoNet. The implementation is based on the abstract base class `MLBaseClass.py`, and each of the algorithms is written in a separated class. The main program is specified in `main.py`. I will try to update for regression soon.

## Operation mechanism explanation
The implementation is mainly in the abstract base class `MLBaseClass.py` with some auxilliary classes and functions in `_utils.py`. The operation principle of the implementation can be divided into 3 steps:

### Step 1: initialize hyper-net and base-net
Recall the nature of the meta-learning as:

&theta; &rarr; __w__ &rarr; y &larr; __x__,

where &theta; denotes the parameter of the hyper-net, __w__ is the base-model parameter, and (__x__, y) is the data.

The implementation is designed to follow this generative process, where the hyper-net will generate the base-net. It can be summarized in the following _pseudo-code_:

```python
# initialization
base_net = ResNet18() # base-net

# convert conventional functional
f_base_net = torch_to_functional_module(module=base_net)

# make hyper-net from the base-net
hyper_net = hyper_net_cls(base_net=base_net)

# the hyper-net generates the parameter of the base-net
base_net_params = hyper_net.forward()

# make prediction
y = f_base_net(x, params=base_net_params)
```
- MAML: the hyper-net is the initialization of the base-net. Hence, the generative process follows identity operator, and hence, `hyper_net_cls` is defined as the class `IdentityNet` in `_utils.py`.
- ABML and VAMPIRE: the base-net parameter is a sample drawn from a diagonal Gaussian distribution parameterized by the meta-parameter. Hence, the hyper-net is designed to simulate this sampling process. In this case, `hyper_net_cls` is the class `NormalVariationalNet` in `_utils.py`.
- Prototypical network is different from the above algorithms due to its metric-learning nature. In the implementation, only one network is used as `hyper_net`, while the `base_net` is set to `None`.

Why is it such a complicated implementation? It is to allow us to share the common procedures of many meta-learning algorithms via the abstract base class `MLBaseClass`. If it is not cleared to you, please open an issue or send me an email. I am happy to discuss to improve the readability of the code further.

### Step 2: task adaptation (often known as inner-loop)
There are 2 sub-functions corresponding to MAML-like algorithms and protonet.

#### `adapt_to_episode` - applicable for MAML-like algorithms
The idea is simple:
1. Generate the parameter(s) of the base-net from the hyper-net
2. Use the generated base-net parameter(s) to calculate loss on _training_ (also known as _support_) data
3. Minimize the loss w.r.t. the parameter of the hyper-net
4. Return the (task-specific) hyper-net (assigned to `f_hyper_net`) for that particular task

#### `adapt to task by calculating prototypes` - applicable for Prototypical Networks
Calculate and return the prototypes in the embedding space

### Step 3: evaluate on validation subset
The task-specific hyper-net, or `f_hyper_net` in the case of MAML-like algorithms, or the prototypes in the case of prototypical networks, are used to predict the labels of the data in the validation subset. 
- In training, the predicted labels are used to calculate the loss, and the parameter of the hyper-net is updated to minimize that loss.
- In testing, the predicted labels are used to compute the prediction accuracy.

Note that ABML is slightly different since it also includes the loss made by the task-specific hyper-net on the training subset. In addition, it places prior on the parameter of the hyper-net. This is implemented in the methods `loss_extra()` and `loss_prior`, respectively.

## Data source
### Regression
The data source for regression is generated from `DataGeneratorT.py`. Currently, regression has not been implemented yet.

### Classification
Initially, Omniglot and mini-ImageNet are the two datasets considered. They are organized following the `torchvision.datasets.ImageFolder`.
```
Dataset
│__alphabet1_character1 (or class1)
|__alphabet2_character2 (or class2)
...
|__alphabetn_characterm (or classz)
```
If this your setup, please modify the code to use `ImageFolderGenerator` as the `eps_generator`.

You can also keep the original structure of Omniglot (train -> alphabets -> characters), and directly run the code without modification. I had this setup in mind when coding the implementation.

For the extracted feature, which I call `miniImageNet_640`, the train-test split is in pickle format. Each pickle file consists of a tuple `all_class, all_data`, where:
- `all_class` is a dictionary where keys are the names of classes, and the values are their corresponding names of images belong to those classes,
- `all_data` is also a dictionary where keys are the names of all images, and values are the vector values of the corresponding images.

You can also download [the resized Omniglot](https://www.dropbox.com/s/w1do3wi0wzzo4jw/omniglot.zip?dl=0) and [the miniImageNet with extracted features](https://www.dropbox.com/s/z48ioy2s2bjbu93/miniImageNet_640.zip?dl=0) from my shared Dropbox. Please note that the extracted features of miniImageNet dataset is done by the authors of LEO nets from DeepMind. The one from my Dropbox is a modified version with scaling and putting into a proper data structure for the ease of use.

## Run
Before running, please go to each script and modify the path to save the files by looking for `logdir`. This is the destination location to save models.

For the legacy implementation such as BMAML, please go to the `utils.py` and modify the default dataset folders (or you can do it by passing the corresponding argument in the `load_dataset` function). The epoch here is locally defined through `expected_total_tasks_per_epoch` tasks (e.g. 10k tasks = 1 epoch), and therefore, different from the definition of epoch in conventional machine learning.

To run, copy and paste the command at the beginning of each algorithm script and change the configurable parameters (if needed).

To test, simply specify which saved model is used via variable `resume_epoch` and replace `--train` by `--test` at the end of the commands found on the top of `main.py`.

## Tensorboard
[Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) is also integrated into the implementation. Hence, you can open it and monitor the training on your favourite browser:
```bash
tensorboard --logdir=<your destination folder>
```
Then open the browser and see the training progress at:
```
http://localhost:6006/
```

## Final note
If you only need to run MAML and feel that my implementation is complicated, [torch-meta](https://github.com/tristandeleu/pytorch-meta) is a worthy repository to take a look. The difference between torch-meta and mine is to extend the implementation to other algorithms, such as VAMPIRE and ABML.

If you feel this repository useful, please give a :star: to motivate my work.

In addition, please consider to give a :star: to the [__higher__][higher repo] repository developed by Facebook. Without it, we still suffer from the arduous re-implementation of model "functional" form.

---
(legacy)

## Test
The command for testing is slightly different from running. This can be done by provide the file (or epoch) we want to test through `resume_epoch=xx`, where `xx` is the id of the file. It is followed by the parameter `--test`, and:
- `--no_uncertainty`: quanlitative result in regression, or output the accuracy of each task in classification,
- `--uncertainty`: outputs a csv-file with accuracy and prediction class probability of each image. This will be then used to calculate the calibration error.

[higher repo]: https://github.com/facebookresearch/higher
