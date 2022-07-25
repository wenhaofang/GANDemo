## GAN Demo

This repository includes some demo GAN models.

Note: The project refers to [YixinChen-AI](https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner) and [eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN)

Datasets:

* `dataset1`: [MNIST](http://yann.lecun.com/exdb/mnist/)

Models:

* `model1`: GAN

* `model2`: WGAN

* `model3`: WGAN-GP

* `model4`: CGAN

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
# WGAN
PYTHONPATH=. python modules/module2.py
# WGAN-GP
PYTHONPATH=. python modules/module3.py
# CGAN
PYTHONPATH=. python modules/module4.py
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

```shell
# module2
python main.py \
    --name 2 \
    --module 2
```

```shell
# module3
python main.py \
    --name 3 \
    --module 3
```

```shell
# module4
python main.py \
    --name 4 \
    --module 4
```

### Note (重点、难点、疑点、TODO、...)

1、WGAN 和 GAN 的区别

* Discriminator 去掉最后一层 Sigmoid
* Optimizer 不使用基于动量的算法，推荐使用 RMSProp、SGD
* Optimizer 更新参数后，将 Discriminator 的参数截断为固定常数的绝对值
* Criterion 计算 loss 不取 log

2、WGAN-GP 和 WGAN 的区别

* 将参数截断替换为梯度惩罚

3、关于 CGAN

* 对于 Discriminator，输入图片加标签，判断是否为真实图片 (二分类)
* 对于 Discriminator，判断：真图片加真标签 (1)，假图片加假标签 (0)

