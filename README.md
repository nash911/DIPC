# DIPC

Double Inverted Pendulum Cart environment with DDPG, PPO, SAC, and TD3.

## Installation

[Poetry](https://python-poetry.org/) is used for dependency management. So please install poetry:

```bash
$ curl -sSL https://install.python-poetry.org | python3 -

```

To install all the dependencies, please enter the following from the project's root directory:

```bash
$ poetry install

```

Then enter the virtual environment:

```bash
$ poetry shell

```

For training the DDPG model:

```bash
$ python ddpg_train.py --verbose --plot

```

For evaluating the training the DDPG model:

```bash
$ python ddpg_eval.py --path <path/to/trained/model/folder>

```

For training the PPO model:

```bash
$ python ppo_train.py --verbose --plot

```

For evaluating the training the PPO model:

```bash
$ python ppo_eval.py --path <path/to/trained/model/folder>

```

For training the SAC model:

```bash
$ python sac_train.py --verbose --plot

```

For evaluating the training the SAC model:

```bash
$ python sac_eval.py --path <path/to/trained/model/folder>

```

For training the TD3 model:

```bash
$ python td3_train.py --verbose --plot

```

For evaluating the training the TD3 model:

```bash
$ python td3_eval.py --path <path/to/trained/model/folder>

```
