## GAN Demo

This repository includes some demo GAN models.

Note: The project refers to [YixinChen-AI](https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner) and [eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN)

Datasets:

* `dataset1`: [MNIST](http://yann.lecun.com/exdb/mnist/)

### Unit Test

* for loaders

```shell
# MNIST
PYTHONPATH=. python loaders/loader1.py
```

* for modules

```shell
# GAN
PYTHONPATH=. python modules/module1.py
```

### Main Process

```shell
python main.py
```

You can change the config either in the command line or in the file `utils/parser.py`

Here are the examples for each module:

```shell
# module1
python main.py \
    --name 1 \
    --module 1
```
