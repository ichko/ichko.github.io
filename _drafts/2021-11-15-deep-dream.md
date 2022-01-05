---
layout: wide-post
title: Deep Dream
subtitle: "Deep Dream in your browser"
date: 2021-11-15 23:15:03 +0200
categories: ml dl deep-dream generators gradient-ascent
comments: true
pinned: true
---

<div class="wrapper" markdown="1">

#### Back propagating neural network impressions

<!-- <script
  src="https://requirejs.org/docs/release/2.3.6/minified/require.js">
</script> -->

<!-- Load TensorFlow.js. This is required to use MobileNet. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis"></script>

<script>
  // const paths = {
  //     tf: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf',
  //     tfModels: 'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0',
  //   }
  // requirejs.config({ paths });

  // requirejs(
  //   Object.keys(paths),
  //   () => window.onload = () => main()
  // );

  window.onload = () => main(tf);

  async function main(tf) {
    const mainDiv = document.getElementById('main');
    tf.randomNormal([4, 4]).print();

    const modelUrl = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_130_224/classification/2/default/1'

    const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true })
    const input = tf.variable(tf.randomNormal([1, 224, 224, 3]));
    const out = model.predict(input);
    const loss = () => model.predict(input).norm();
    const optim = tf.train.sgd(0.1);
    const {grads} = optim.computeGradients(loss, [input])

    const canvas = document.createElement('canvas');
    canvas.width = input.shape.width;
    canvas.height = input.shape.height;

    // const image = input.gather(0).clipByValue(0, 1);
    let image = grads[0];
    image = image.gather(0);
    image = image.add(image.min().neg());
    // image = image.clipByValue(0, 1);

    debugger;
    await tf.browser.toPixels(image, canvas);
    mainDiv.appendChild(canvas)

    // mobilenet.load().then(model => {
    //   // Classify the image.
    //   debugger;
    //   model.classify(img).then(predictions => {
    //     console.log('Predictions: ');
    //     console.log(predictions);
    //   });
    // });
  }
</script>

<div id="main"></div>

<!-- ## GAN in a single dimension

![1D Perceptrons](../assets/one-d-gan/1d-perceptrons.svg) -->

<!-- ## Mode collapse

test

<img
  class="center-image"
  style="border: 3px solid #eee"
  src="/assets/one-d-gan/mode-collapse.svg"
/>

<div class="fig" markdown="1">
  **Fig. 2:** 1D Mode collapse.
</div>

<video class="center-image" controls autoplay="autoplay" loop="">
  <source src="https://ichko.github.io/ml-playground/notebooks/distribs3.webm">
  Your browser does not support the video tag.
</video>

## Resources and Tools

- [Test](#) -->

</div>
