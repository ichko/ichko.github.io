---
layout: post
title: "Tutorial: Visualizing VAE with Efemarai"
date: 2020-11-13 13:38:03 +0200
categories: ml dl tutorial vae ae auto-encoder
comments: true
---

<img src="/assets/vae-efem-tutorial/efem-train.gif" class="center-image hundred-width styled-border" />

In this tutorial, we are going to implement and visualize the training process of **Variational Auto-Encoder**.

## What is VAE

### Auto-Encoder

To explain what a VAE is we must first explain the general architecture of an auto-encoder. Auto-encoders are are a type of neural network architecture that is designed to do dimensionality reduction. The network is a composition of an encoder and a decoder. The job of the encoder is to take an input and “compress” it to a vector with less dimensionality. The job of the decoder is to take this lower dimensionality representation and expand it to the original shape of the data. The auto-encoder is then trained in unsupervised manned to output whatever data we input into it. The point of this is for the encoder to learn to squeeze the original data in vectors with lower dimensions preserving only the most important features of the data - the ones that can then be used to reconstruct the data with higher precision by the decoder.

<object class="center-image hundred-width" type="image/svg+xml" data="/assets/vae-efem-tutorial/ae.svg">
  SVG not supported
</object>

### Variational Auto-Encoder

After seeing what an AE is let’s see how the variation of a VAE comes into play.
Let’s say we want to generate new data after we have trained our auto-encoder. We can do that by sampling latent vectors and passing them to the decoder, but we don’t know the distribution of the latent dimension. We don’t know how to get latent vectors that are going to be decoded into meaningful outputs.
This is what the VAE paper addresses. With a VAE instead of learning to encode a fixed vector from which to decode, we learn to encode the parameters of a normal distribution. Using a KL term in the loss function we can force the distribution to be a multivariate normal distribution with mean zero and variance one.

KL stands for _Kullback–Leibler_ divergence which is a measure of distance between two distributions. Minimizing the KL loss in addition to the reconstruction loss lets us have an auto-encoder with known (normal) latent distribution, which we can then sample from and generate new data points by decoding the sampled vectors.

<img src="/assets/vae-efem-tutorial/vae.svg" class="center-image hundred-width" />

## Implementation

The implementation will be done in [Pytorch](https://pytorch.org/). We start by creating a torch module.

```py
class VAE(nn.Module):
    def __init__(self, input_shape, encoding_size):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        flatten_size = np.prod(list(input_shape))
        encoding_size = encoding_size

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_size * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, flatten_size),
            nn.Sigmoid(),
        )

        self.loss = nn.BCELoss(reduction='mean')

    def forward(self, x):
      pass
```
