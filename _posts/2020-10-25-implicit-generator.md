---
layout: post
title: Implicit generator
date: 2020-10-25 13:38:03 +0200
categories: ml dl neural-networks generators
comments: true
---

## 🚧 Under development

### VAE as augmentation function

For a final experiment lets do the most generic setup possible. Up until now we trained
the `generator` to generate images robust to particular disruptions, which we deemed
innate properties of natural images. What if we try to do something even more generic.
At the end of the day we want images with structure. And something that has structure has low
entropy, meaning it can be compressed. Which neural network architecture can compress?
An `AutoEncoder`, a normal one. Let's replace our disruption function with an `AE` and
force the generator to generate images that are compressible (by an `AE`) - have low entropy.

More precisely the network we will experiment with look have the following structure:

![Diagram of Hourglass network](/assets/inverted-ae/hourglass-network.png)

What will the structure of these _expanding_ and _squeezing_ networks be. Well,
if they are stacks of dense layers the property of local dependence of the pixels will be lost,
because the network would not differentiate between pixels that are near by vs pixels that are far apart.
But if we use `Conv2D` layers, because of the nature of the convolution operation,
we can expect structures with local dependence.
