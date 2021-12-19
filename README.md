Official code for [Continual Learning In Environments With Polynomial Mixing Times](https://arxiv.org/abs/2112.07066)

# Continual Learning in Environments with Polynomial Mixing Times
This repository provides official code base for the paper "Continual Learning in Environments with Polynomial Mixing Times"

## Basic Setup
Clone this repository and then follow this command

```cd polynomial-mixing-times```

Create either use a python virtualenv or a conda environment and activate it. 

```
pip install virtualenv
virtualenv -p /usr/bin/python3.7 mixing-times
source mixing-times/bin/activate
```

To install all the relevant packages use the following command:

```pip install -e .```

## Running the experiments
We provide a running script with all relevant hyperparameters used for both baselines and our proposed model. One can run ``run_bottleneck.sh`` to run all the models.

To run the experiments of the proposed models on the Example 2 Bottleneck MDP class with 4 rooms, "random" task evolution and a random seed of 1, use the following command
```
bash run_bottleneck.sh 1 4 "random"
```
### Available Models
1. Online Q learning
2. Q learning with Replay
3. Q learning w/ Dyna
4. Model based n-step TD
5. Vanilla Policy Gradient
6. Onpolicy rho learning
7. Off-policy rho learning
8. rho Policy Gradient

### List of Environments
1. ``ScaleClass-v0``
2. ``NBottleneckClass-v0``
3. ``NCycleClass-v0``

### System requirements
We used python 3.7 version to run all our experiments. 


