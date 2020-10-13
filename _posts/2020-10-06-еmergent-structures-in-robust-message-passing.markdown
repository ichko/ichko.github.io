---
layout: post
title: Emergent structures in noisy channel message-passing
date: 2020-10-06 13:38:03 +0200
categories: ml dl philosophy auto-encoders neural networks
comments: true
---

So recently I stumbled upon this interesting repository - [noahtren/GlyphNet](https://github.com/noahtren/GlyphNet), explaining a mechanism for generating glyphs, like the glyphs we use in human languages:

<img class="center-image" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Rosetta_Stone.JPG/280px-Rosetta_Stone.JPG" alt="drawing" width="350"/>

<div class="fig">
  Source: <a href="https://en.wikipedia.org/wiki/Rosetta_Stone">Rosetta Stone Wikipedia</a>
</div>

These are images with 2D structure used to communicate information. We will use neural networks to generate those structured images. The basic idea is rather simple and the principles at play can be explained easily.

We want to generate images, containing visuals information, much like the images from the MNIST dataset. So what constitutes something that can be recognized in an image? Well, images are mostly smooth in their 2D visual representation - meaning pixels that are close to each other tend to have a smaller difference between their values and to be highly correlated. Also, images are robust to some sort of perturbations, visual noise, translations, rotations, and more. Meaning the information contents of an image is mostly preserved under those transformations.
So what can we do if we want to generate images with these properties without any sort of dataset? We can optimize for robustness. But what do I mean by that?

For our experiment, we will have the following setup:

![Diagram of message passing](/assets/inverted-ae/diagram-of-message-passing.png)

- So what we have is a message, being randomly generated at the beginning.
- We pass this message through an encoder (possibly conv-transpose multi-layer network).
- We induce differentiable noise into the image - this means that we transform the image trough a differentiable function. Examples of such functions include:
  - Arbitrary UV-remap (being differentiable with the use of spatial transformer)
    - Translation
    - Rotation
    - Crop
    - etc.
  - Multiplying random noise
  - And other transformation which we would like the generated images to be robust under
    We then pass the disrupted image through a decoder (possibly conv multi-layer network) resulting in vector.
  - We optimize the output to be the same as the input .

In a way, we are optimizing the generator to generate images that can be understood even if perturbed slightly. Depending on the noise under which we optimize the generator learns to induce robustness into the images. It adds patterns with high dimensionality which are used to communicate the information continued in the input message.
As a neural network architecture, this looks like an **Inverted Auto-Encoder** and it is trained exactly like a regular **AE**.

![Diagram of the model](/assets/inverted-ae/diagram-of-inverted-ae.png)

The networks are communicating normally distributed points from ${\Bbb R}^{N}$, where $N$ is the size of the input message.

The model is the same as an Auto-Encoder, it just reverses the positions of the `compressor`, commonly known as **Encoder**, but here it is named **Decoder** since it decodes the initial message, and the `generator`,
commonly known as **Decoder**, but here it is named **Generator** since it generates the image.
The generator has to learn to generate images that are invariant in their information contents after they are augmented.

<!-- The second law of thermodynamics states that “Isolated systems spontaneously evolve towards thermodynamic equilibrium, the state with maximum entropy”, meaning that nature is pushing the world towards being noisier and noisier and humans have naturally evolved means to exchange information in a robust manner. -->

Looking at this we may start to think more and more about the nature of everything we call structured. Is robustness innate property of anything we deem structured? Maybe not, but what about structure used to communicate information between humans - like pictures or natural language.

## Generating structure from nothing

To define and train the model we will be using [PyTorch](https://pytorch.org/), modern and powerful library for all things _Deep Learning_.
To define the differentiable noise function we will use [Kornia](https://kornia.github.io/), because it has useful computer vision functions commonly used in data augmentation.

For the model definition you can see the code in this repo [inverted-ae](/todo). You can inspect it, it is nothing special,
just a few activated and batch normalized `ConvTranspose2d` layer for the `Generator` and activated and batch normalized `Conv2d`
layers for the `Message Reconstruction` module.

Between them we place sequence of `kornia` random augmentation differentiable modules like so:

```python
img = load_img(url, size=512)          # Load
img = img.unsqueeze(0).float() / 255   # Normalize
imgs = T.cat(16 * [img])               # Add multiple in batch
normal_noise = T.randn_like(imgs) / 4  # Generate random noise

noise = nn.Sequential(
    kornia.augmentation.RandomAffine(
        degrees=30,
        translate=[0.1, 0.1],
        scale=[0.9, 1.1],
        shear=[-10, 10],
    ),
    kornia.augmentation.RandomPerspective(0.6, p=0.5),
)

augmented_imgs = noise(imgs + normal_noise)

imshow(augmented_imgs)
```

This definition of the augmentation function can yields results like the following:

<img class="center-image" src="/assets/inverted-ae/kit-cat-augmentation.png" alt="Augmented kit-cat" />

As we can see a we have a lot of affine variation, as well as shift of perspective. The normal noise added before the augmentation
is also an important component as it leads to the generator learning to generate smoother images.

This is actually all we need in terms of model definition. You can look at the code in the repo for more in dept code example.

For the data we will be using the following generator:

```python
msg_size = 64

def sample(bs):
    return T.randn(bs, msg_size).to(DEVICE)

def get_data_gen(bs):
    while True:
        X = sample(bs)
        yield X, X
```

Really simple stiff. We generate randomly distributed vector with `msg_size` dimensions and we yields tuples of this vector replicated as we will train
our model just like an `Auto-Encoder`.

We are ready to toss the model and the data generator in our favorite `PyTorch` training loop framework. In my case that is a simple home brewed library
that is part of the code in the repository of the project.

**Lets fit and plot the generated images as we train!**

<video class="center-image" controls loop="">
  <source src="/assets/inverted-ae/inverted-ae-training.webm">
  Your browser does not support the video tag.
</video>

<div class="fig">
  Images generated from the same initial messages during training.
</div>

<div class="fig">
  <img class="center-image" src="/assets/inverted-ae/inverted-ae-trained.png" alt="Trained Inverted AE" />
  The final result of the generated images from initial message sample.
</div>

If we try to cram the same amount of information - normally distributed vectors with size 64 into a `100x100` images we get the following result.

![Diagram of the model](/assets/inverted-ae/64to100x100.png)

Not bad, but not particularly interesting if you ask me.
Let's now try to progressively increase the message size and view how that changes the generated images. The idea is that this might force the
generator to generate objects with higher level of detail.

#### Lets up the resolution

#### Adding colors

#### Interpolating in the latent space

#### Encoding and decoding MNIST

#### VAE as augmentation function
