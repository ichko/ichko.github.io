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
    tf.randomNormal([4, 4]).print();

    const model = await tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_140_224/classification/1/default/1", { fromTFHub: true })
    debugger;
    console.log('Mobilenet model is loaded')

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
