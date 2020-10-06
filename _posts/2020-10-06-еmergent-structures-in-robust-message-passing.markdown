---
layout: post
title: Emergent structures in noisy channel message-passing
date: 2020-01-04 13:38:03 +0200
categories: ml, dl, philosophy, auto encoders, neural networks
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
    We then pass the disrupted image through a decoder (possibly conv multi-layer network) resulting in vector
    We optimize the output to be the same as the input .

In a way, we are optimizing the generator to generate images that can be understood even if perturbed slightly. Depending on the noise under which we optimize the generator learns to induce robustness into the images. It adds patterns with high dimensionality which are used to communicate the information continued in the input message.
As a neural network architecture, this looks like an **Inverted Auto-Encoder** and it is trained exactly like on.

![Diagram of the model](/assets/inverted-ae/diagram-of-inverted-ae.png)

<div class="fig">
  Diagram of Inverted Auto-Encoder. The actual model would not generate images looking like the images in the MNIST dataset.
</div>

The model is the same as an Auto-Encoder, it just reverses the positions of the `compressor` and the `generator`.

Looking at this we may start to think more and more about the nature of everything we call structured. Is robustness innate property of anything we deem structured? Maybe not, but what about structure used to communicate information between humans - like pictures or natural language.

The second law of thermodynamics states that “Isolated systems spontaneously evolve towards thermodynamic equilibrium, the state with maximum entropy”, meaning that nature is pushing the world towards being noisier and noisier and humans have naturally evolved means to exchange information in a robust manner.

The networks are communicating normally distributed points from ${\Bbb R}^{N}$, where $N$ is the size of the input message.

## Generating structure from nothing

To define and train the model we will be using [PyTorch](https://pytorch.org/)`, modern and powerful library for all things _Deep Learning_.
To define the differentiable noise function we will use [Kornia](https://kornia.github.io/), because it has useful computer vision functions commonly used in data augmentation.

<video class="center-image" autoplay="autoplay" loop="">
  <source src="/assets/inverted-ae/reverse-ae-training.webm">
  Your browser does not support the video tag.
</video>

<div class="fig">
  Images generated from the same initial messages during training.
</div>

TODO...
