---
layout: post
title:  "Generating random art with neural networks"
date:   2019-11-15 21:39:01 +0200
categories: jekyll update
image: /assets/cppn/0_colorful_out.png
---

![colorful output from the cppn](/assets/cppn/1_cool_fire.png)

Recently, I discovered [blog.otoro.net](http://blog.otoro.net/) - David Ha's blog
and I was delighted to see all kinds of creative applications of deep learning technics.
In this post I will attempt to replicate and build on top of 
[his work](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/) on
[CPPNs](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network).
As this post is heavily inspired by the articles of the blog mentioned above
I strongly recommend that you read them as well.

## What is CPPN

First things first, what is a `Compositional Pattern Producing Network`.
The Wikipedia definition is as follows:

> ANNs that have an architecture whose evolution is guided by genetic algorithms.

Well, in the current post I would not be doing anything related to genetic algorithms.
The architecture of the networks would not be evolving and neither is the topology
of the computation be anything more than a vanilla feed forward network
as the FFN would be enough to produce interestingly looking results.

What the network is actually going to do is take a discrete 2D vector field
(mesh grid) and map it to the 3D space of colors.

$$
  f: {\Bbb R}^2 \to {\Bbb R}^3
$$

Since the input mesh grid will be somewhat smooth as viewed in a 2D matrix
and because the neural network is [continuous function](https://en.wikipedia.org/wiki/Continuous_function),
we would expect the results to resemble radom, but smooth transitions between colors.
In a sense the neural network would act as a `fragment shader` just like the one you
have in [Shadertoy](http://shadertoy.com). Taking in the `uv` coordinates of the pixels
and mapping them to colors.

$$
  f(u, v) = \sigma(...W_2 a(W_1 \vec{uv} + b_1) + b_2...)
$$

Here $a$ is some activation function (like $tanh$ or $\sigma$) and $\sigma$ is the sigmoid function.
We naturally activate the last layer with $\sigma$ so that we map of the output is in the range $(0, 1)$.

![function mapping trough the layers](/assets/cppn/5_transition.png)

Since the network would not be trained the output would depend on the random
initialization of all the parameters of the network hence - generating _random art_.
We would extend this by adding a random (latent) vector as an input which would vary the generated image.
The vector would be fixed for the pixels of a single image but varied in time leading to a smooth transitions
in time.

<video width="100%" autoplay="autoplay" loop>
  <source src="/assets/cppn/vid_1.webm">
  Your browser does not support the video tag.
</video>

# Some examples
