# 生成对抗性神经网络

**[论文](https://arxiv.org/pdf/1406.2661v1.pdf)**

**作者: [Lorna](https://github.com/Lornatang)**

**邮箱: [shiyipaisizuo@gmail.com]()**

[英文原版](https://github.com/Lornatang/Generative-Adversarial-Networks/blob/master/README.md)

**配置需求**

- 显卡: A TiTAN V 或更高.
- 硬盘: 128G SSD.
- Python版本： python3.5 或更高.
- CUDA: cuda10.
- CUDNN: cudnn7.4.5 或更高.
- Tensorflow-gpu: 2.0.0-alpla0.

*运行以下代码。*
```text
pip install -r requirements.txt
```

# GAN是什么?
生成对抗网络(GANs)是当今计算机科学中最有趣的概念之一。
两个模型通过对抗性过程同时训练。
生成器(“艺术家”)学会创建看起来真实的图像，而鉴别器(“艺术评论家”)学会区分真实图像和赝品。
![](https://github.com/Lornatang/Generative-Adversarial-Networks/blob/master/imgs/intro.png)

在训练过程中，生成器逐渐变得更擅长创建看起来真实的图像，而鉴别器则变得更擅长区分它们。
当鉴别器无法分辨真伪图像时，该过程达到平衡。
![](https://github.com/Lornatang/Generative-Adversarial-Networks/blob/master/imgs/train.png)

下面的动画展示了生成器在经过50个时代的训练后生成的一系列图像。
这些图像一开始是随机噪声，随着时间的推移越来越像手写数字。
![](https://github.com/Lornatang/Generative-Adversarial-Networks/blob/master/imgs/dcgan.gif)

## 1.介绍

### 1.1 原理
这是一张关于GAN的流程图
<div align=center>![GAN](https://github.com/Lornatang/Generative-Adversarial-Networks/blob/master/imgs/GAN.png)

GAN主要的灵感来源是零和游戏在博弈论思想,应用于深学习神经网络,是*通过生成网络G(发电机)和判别D(鉴频器)网络游戏不断,从而使G*学习数据分布,如果用在图像生成训练完成后,G可以从一个随机数生成逼真的图像。
G和D的主要功能是:

- G是一个生成网络，它接收一个随机噪声z(随机数)，通过噪声生成图像。

- D是一个判断图像是否“真实”的网络。它的输入参数是x, x代表一张图片，输出D (x)代表x是一张真实图片的概率。如果是1，代表100%真实的图像，如果是0，代表不可能的图像。

在训练过程中，生成网络G的目标是生成尽可能多的真实图像来欺骗网络D，而D的目标是试图将G生成的假图像与真实图像区分开来。这样，G和D构成一个动态的“博弈过程”，最终的均衡点为纳什均衡点。

### 1.2 体系结构
通过对目标的优化，可以调整概率生成模型的参数，使概率分布与实际数据分布尽可能接近。

那么，如何定义适当的优化目标或损失呢?
在传统的生成模型中，一般采用数据的似然作为优化目标，而GAN创新性地使用了另一个优化目标。

- 首先，引入判别模型(常用模型包括支持向量机和多层神经网络)。

- 其次，其优化过程是在生成模型和判别模型之间找到纳什均衡。

GAN建立的学习框架实际上是生成模型和判别模型之间的模拟博弈。
生成模型的目的是尽可能多地模拟、建模和学习真实数据的分布规律。
判别模型是判断一个输入数据是来自真实的数据分布还是生成的模型。
通过这两个内部模型之间的持续竞争，提高了生成和区分这两个模型的能力。

当一个模型具有很强的区分能力时。
*如果生成的模型数据仍然存在混淆，不能正确判断，那么我们认为生成的模型实际上已经了解了真实数据的分布情况。*

### 1.3 GAN特性
**特点:**

- low与传统模式相比，有两种不同的网络，而不是单一的网络，采用的是对抗训练方法和训练方式。

- 更新信息中的低GAN梯度G来自判别式D，而不是来自样本数据。

**优势:**

- low GAN是一个涌现模型，相对于其他生成模型(玻尔兹曼机和GSNs)，它只通过反向传播，不需要复杂的马尔可夫链。

- 与其它所有机型相比，GAN能生产出更清晰、真实的样品

- low GAN是一种无监督学习训练，可广泛应用于半监督学习和无监督学习领域。

- 与变分自编码器相比，GANs不引入任何确定性偏差，变分方法引入确定性偏差，因为它们优化了对数似然的下界而不是似然本身，这似乎导致VAEs生成的实例比GANs更加模糊。

- 与VAE、GANs的变分下界相比较低，如果判别器训练良好，则生成器可以学习完善训练样本分布。换句话说，GANs是逐渐一致的，但是VAE是有偏见的。

- GAN——应用于一些场景,比如*图片风格转换、超分辨率,图像完成、噪声去除*,避免了损失函数设计的困难,只要有一个基准,直接鉴别器,其余的对抗训练。

**缺点:**

- 训练GAN需要达到Nash均衡，有时可以通过梯度下降法实现，有时则不能。我们还没有找到一个很好的方法来达到纳什均衡，所以与VAE或PixelRNN相比，GAN的训练是不稳定的，但我认为在实践中它比训练玻尔兹曼机更稳定。

- GAN不适用于处理离散数据，如文本。

- GAN存在训练不稳定、梯度消失和模态崩溃等问题。

## 2. 实现
加载和准备数据集,将使用MNIST数据集来训练生成器和鉴别器。生成器将生成类似MNIST数据的手写数字。

```python
import tensorflow as tf


def load_dataset(mnist_size, mnist_batch_size, cifar_size, cifar_batch_size,):
  """ load mnist and cifar10 dataset to shuffle.

  Args:
    mnist_size: mnist dataset size.
    mnist_batch_size: every train dataset of mnist.
    cifar_size: cifar10 dataset size.
    cifar_batch_size: every train dataset of cifar10.

  Returns:
    mnist dataset, cifar10 dataset

  """
  # load mnist data
  (mnist_train_images, mnist_train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

  # load cifar10 data
  (cifar_train_images, cifar_train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

  mnist_train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1).astype('float32')
  mnist_train_images = (mnist_train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

  cifar_train_images = cifar_train_images.reshape(cifar_train_images.shape[0], 32, 32, 3).astype('float32')
  cifar_train_images = (cifar_train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

  # Batch and shuffle the data
  mnist_train_dataset = tf.data.Dataset.from_tensor_slices(mnist_train_images)
  mnist_train_dataset = mnist_train_dataset.shuffle(mnist_size).batch(mnist_batch_size)

  cifar_train_dataset = tf.data.Dataset.from_tensor_slices(cifar_train_images)
  cifar_train_dataset = cifar_train_dataset.shuffle(cifar_size).batch(cifar_batch_size)

  return mnist_train_dataset, cifar_train_dataset
```

### 2.2 创造模型文件
生成器和鉴别器都使用[Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

#### 2.2.1 生成器模型
这里只对神经网络体系结构使用最基本的全连接形式。
除第一层不使用归一化外，其余层均由全连接的线性结构定义——>归一化——>LeakReLU，具体参数如下所示。

```python
import tensorflow as tf
from tensorflow.python.keras import layers


def make_generator_model(dataset='mnist'):
  """ implements generate.

  Args:
    dataset: mnist or cifar10 dataset. (default='mnist'). choice{'mnist', 'cifar'}.

  Returns:
    model.

  """
  model = tf.keras.models.Sequential()
  model.add(layers.Dense(256, input_dim=100))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(512))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(1024))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.2))

  if dataset == 'mnist':
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
  elif dataset == 'cifar':
    model.add(layers.Dense(32 * 32 * 3, activation='tanh'))
    model.add(layers.Reshape((32, 32, 3)))

  return model
```

#### 2.2.2 鉴别器模型
鉴别器是一种基于cnn的图像分类器。

```python
import tensorflow as tf
from tensorflow.python.keras import layers


def make_discriminator_model(dataset='mnist'):
  """ implements discriminate.

  Args:
    dataset: mnist or cifar10 dataset. (default='mnist'). choice{'mnist', 'cifar'}.

  Returns:
    model.

  """
  model = tf.keras.models.Sequential()
  if dataset == 'mnist':
    model.add(layers.Flatten(input_shape=[28, 28, 1]))
  elif dataset == 'cifar':
    model.add(layers.Flatten(input_shape=[32, 32, 3]))

  model.add(layers.Dense(1024))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(512))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(256))
  model.add(layers.LeakyReLU(alpha=0.2))

  model.add(layers.Dense(1, activation='sigmoid'))

  return model
```

### 2.3 定义损失和优化器

### 2.3.1 为这两个模型定义损失函数。

```text
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```
#### 2.3.2 鉴别器损失函数
该方法量化了鉴别器对真伪图像的识别能力。
它将鉴别器对真实图像的预测与1的数组进行比较，将鉴别器对假(生成的)图像的预测与0的数组进行比较。

```text
def discriminator_loss(real_output, fake_output):
  """ This method quantifies how well the discriminator is able to distinguish real images from fakes.
      It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions
      on fake (generated) images to an array of 0s.

  Args:
    real_output: origin pic.
    fake_output: generate pic.

  Returns:
    real loss + fake loss

  """
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss

  return total_loss
```

#### 2.3.3 生成器损失函数
发电机的损耗量化了它欺骗鉴别器的能力。
直观地说，如果生成器运行良好，鉴别器将把假图像分类为真实图像(或1)。
在这里，我们将把鉴别器对生成图像的判断与1的数组进行比较。

```text
def generator_loss(fake_output):
  """ The generator's loss quantifies how well it was able to trick the discriminator.
      Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1).
      Here, we will compare the discriminators decisions on the generated images to an array of 1s.

  Args:
    fake_output: generate pic.

  Returns:
    loss

  """
  return cross_entropy(tf.ones_like(fake_output), fake_output)
```

#### 2.3.4 优化
由于我们将分别训练两个网络，因此鉴别器和生成器优化器是不同的。

```text
def generator_optimizer():
  """ The training generator optimizes the network.

  Returns:
    optim loss.

  """
  return tf.keras.optimizers.Adam(lr=1e-4)


def discriminator_optimizer():
  """ The training discriminator optimizes the network.

  Returns:
    optim loss.

  """
  return tf.keras.optimizers.Adam(lr=1e-4)
```

### 2.4 保存训练模型
本笔记本还演示了如何保存和恢复模型，这在长时间运行的训练任务被中断时是很有帮助的。

```python
import os
import tensorflow as tf


def save_checkpoints(generator, discriminator, generator_optimizer, discriminator_optimizer, save_path):
  """ save gan model

  Args:
    generator: generate model.
    discriminator: discriminate model.
    generator_optimizer: generate optimizer func.
    discriminator_optimizer: discriminator optimizer func.
    save_path: save gan model dir path.

  Returns:
    checkpoint path

  """
  checkpoint_dir = save_path
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  return checkpoint_dir, checkpoint, checkpoint_prefix
```

### 2.5 训练

#### 2.5.1 训练设置
训练循环从生成器接收随机种子作为输入开始。
种子是用来产生图像的。
然后使用鉴别器对真实图像(来自训练集)和伪造图像(由生成器生成)进行分类。
计算了每一种模型的损耗，并利用梯度对产生器和鉴别器进行了更新。

```python
from dataset.load_dataset import load_dataset
from network.generator import make_generator_model
from network.discriminator import make_discriminator_model
from util.loss_and_optim import generator_loss, generator_optimizer
from util.loss_and_optim import discriminator_loss, discriminator_optimizer
from util.save_checkpoints import save_checkpoints
from util.generate_and_save_images import generate_and_save_images

import tensorflow as tf
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str,
                    help='use dataset {mnist or cifar}.')
parser.add_argument('--epochs', default=50, type=int,
                    help='Epochs for training.')
args = parser.parse_args()
print(args)

# define model save path
save_path = 'training_checkpoint'

# create dir
if not os.path.exists(save_path):
  os.makedirs(save_path)

# define random noise
noise = tf.random.normal([16, 100])

# load dataset
mnist_train_dataset, cifar_train_dataset = load_dataset(60000, 128, 50000, 64)

# load network and optim paras
generator = make_generator_model(args.dataset)
generator_optimizer = generator_optimizer()

discriminator = make_discriminator_model(args.dataset)
discriminator_optimizer = discriminator_optimizer()

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoints(generator,
                                                                 discriminator,
                                                                 generator_optimizer,
                                                                 discriminator_optimizer,
                                                                 save_path)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  """ break it down into training steps.

  Args:
    images: input images.

  """
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss,
                                             generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                  discriminator.trainable_variables)

  generator_optimizer.apply_gradients(
    zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(
    zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  """ train op

  Args:
    dataset: mnist dataset or cifar10 dataset.
    epochs: number of iterative training.

  """
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    generate_and_save_images(generator,
                             epoch + 1,
                             noise,
                             save_path)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch+1} is {time.time()-start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           noise,
                           save_path)


if __name__ == '__main__':
  if args.dataset == 'mnist':
    train(mnist_train_dataset, args.epochs)
  else:
    train(cifar_train_dataset, args.epochs)
```

### 2.6 生成图片并保存

```python
from matplotlib import pyplot as plt


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  plt.close(fig)
```

## 3. 常见问题

### 3.1 为什么GAN中的优化器不经常使用SGD

- SGD易动摇，易使GAN训练不稳定。

- GAN的目的是在高维非凸参数空间中寻找纳什均衡点。

GAN的纳什均衡点是鞍点，而SGD只会找到局部最小值，因为SGD解决了寻找最小值的问题，而GAN是一个博弈问题。

### 3.2为什么GAN不适合处理文本数据

文本数据是离散的图像数据相比,因为文本,通常需要地图一个词作为一个高维向量,最后预测输出是一个热向量,假设softmax输出(0.2,0.3,0.1,0.2,0.15,0.05)就变成了onehot, 1, 0, 0, 0, 0(0),如果将softmax输出(0.2,0.25,0.2,0.1,0.15,0.1),一个仍然是热(0,1,0,0,0,0)。
因此，对于生成器，G输出不同的结果，而D给出相同的判别结果，不能很好地将梯度更新信息传递给G，因此D最终输出的判别是没有意义的。

- 此外，GAN的损失函数为JS散度，不适合测量不想相交的分布之间的距离。

### 3.3 GAN的一些技能培训

- 使用tanh begin将输入规范化为(- 1,1)，最后一级激活函数(异常)

- 使用wassertein GAN的loss函数，

- 如果你有标签数据，尝试使用标签。有人建议使用倒装标签，并使用标签平滑，单侧标签平滑或双侧标签平滑

- 使用小型批处理范数，如果不使用批处理范数，可以使用实例范数或权重范数

- 避免使用RELU和池化层来降低稀疏梯度的可能性，可以使用leakrelu激活函数

- 优化器尽可能选择ADAM，学习速度不应该太大。初始的1e-4可以参考。此外，随着培训的进行，学习率可以不断降低。

- 在D的网络层中加入高斯噪声相当于一种正则化

### 3.4 模型崩溃原因
一般来说，GAN在训练中并不稳定，效果也很差。
然而，即使延长培训时间，也不能很好地改善。

具体原因可以解释如下:
是针对甘用的训练方法，G梯度更新自D，生成的G很好，所以D对我说什么。
具体来说，G将生成一个样本，并将其交给D进行评估。
D将输出生成的假样本为真样本的概率(0-1)，这相当于告诉G生成的样本有多真实。
G会根据这个反馈改进自身，提高D输出的概率值。
但如果G生成样本可能不是真的,但是D给予正确的评价,或者是G的结果生成的一些特征的识别D,然后G输出会认为我是对的,所以我所以输出D肯定也会给一个高评价,G实际上不是生成的,但他们是两个自我欺骗,导致最终的结果缺少一些信息,特征。

## 4. GAN在生活中的应用

- GAN本身就是一个生成模型，所以数据生成是最常见的，最常见的是图像生成，常用的DCGAN WGAN开始，个人感觉在开始时最好也最简单。

- *GAN本身也是一个无监督学习的模型*。因此在无监督学习和半监督学习中得到了广泛的应用。

- GAN不仅在生成领域发挥作用，还在分类领域发挥作用。简而言之，它是将识别器替换为分类器，执行多个分类任务，而生成器仍然执行生成任务并辅助分类器训练。

- *GAN可以与强化学习*相结合。seq-gan就是一个很好的例子。

- 目前，GAN在*图像样式转换、图像降噪和恢复、图像超分辨率*等方面都有很好的应用前景。


## 待办事项

- 编写FID代码。

— 创建GIF。

# 致谢
[Sakura55](https://blog.csdn.net/Sakura55/article/details/81512600)
