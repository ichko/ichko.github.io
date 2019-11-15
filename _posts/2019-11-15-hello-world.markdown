---
layout: post
title:  "Generating random art with neural networks"
date:   2019-11-15 21:39:01 +0200
categories: jekyll update
---
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/0.9.0/p5.min.js"></script>

<script>
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // Generate some synthetic data for training.
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  // Train the model using the data.
  model.fit(xs, ys, {epochs: 10}).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    model.predict(tf.tensor2d([5], [1, 1])).print();
    // Open the browser devtools to see the output
  });
</script>

Recently, I discovered [blog.otoro.net](http://blog.otoro.net/) - David Ha's blog and I was delighted to sell all kinds of creative applications of deep learning technics.
In this post I will attempt to replicate and build on top of [his work](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/) on [CPPNs](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network).
I will be building my art generating networks in [TensorFlow.js](https://www.tensorflow.org/js), since I've been needing an excuse to try it out in a long time.

# TODO
 - Explain CPPN's
 - Explain why the output is structured - smoothness
 - Relation to shaders
 - Maragoni effect
 - Transformations of pixel data (color to color)
 - Transformations of uv map
 - Latent variable interpolation gifs with the ideas on top
