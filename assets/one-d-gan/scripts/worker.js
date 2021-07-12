importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs");
importScripts("/assets/one-d-gan/scripts/gan.js");

function main(OneDGAN, data, tf) {
  const generateData = data.bimodal(0.3);
  const inpDims = 1;

  const gan = new OneDGAN(0.005, inpDims);

  const targetData = generateData(512);
  const targetDataSync = targetData.dataSync();
  const fixedInputs = gan.sampleZ([512, inpDims]);
  const fixedInputSync = fixedInputs.dataSync();
  const fakeOutputs = gan.G.predict(fixedInputs);
  const fakeOutputsSync = fakeOutputs.dataSync();

  const rangeInputs = tf.range(-3, 3, 0.1).expandDims(1);
  const rangeInputSync = rangeInputs.dataSync();

  // This is done for optimization (single pass of the generator)
  const combinedGInput = fixedInputs.concat(rangeInputs);
  const combinedDInput = rangeInputs.concat(fakeOutputs).concat(targetData);

  self.postMessage({
    type: "init",
    payload: {
      rangeInputSync,
      fakeOutputsSync,
      targetDataSync,
      fixedInputSync,
      fPredictRangeInput: gan.G.predict(rangeInputs).dataSync(),
      dPredictRangeInput: gan.D.predict(rangeInputs).dataSync(),
      dPredictTargetData: gan.D.predict(targetData).dataSync(),
      dPredictFakeData: gan.D.predict(fakeOutputs).dataSync(),
    },
  });

  let i = 0;
  onmessage = async function ({ data }) {
    if (data.type == "optim-step:request") {
      const bs = i == 0 ? 2 : 128;
      const batch = generateData(bs);

      const loss = await gan.optimStep(batch);
      const [gLoss, dLoss] = loss;

      const combinedGOutput = gan.G.predict(combinedGInput);
      const [fakeOutputsSync, ganRangeOutputSync] = [
        combinedGOutput.slice([0], [fixedInputs.shape[0]]).dataSync(),
        combinedGOutput
          .slice([fixedInputs.shape[0]], [rangeInputs.shape[0]])
          .dataSync(),
      ];

      const combinedDOutput = gan.D.predict(combinedDInput);
      const [ganRangeDOutputSync, dFakeOutputsSync, dTargetDataSync] = [
        combinedDOutput.slice([0], [rangeInputs.shape[0]]).dataSync(),
        combinedDOutput
          .slice([rangeInputs.shape[0]], [fakeOutputs.shape[0]])
          .dataSync(),
        combinedDOutput
          .slice([fakeOutputs.shape[0]], [targetData.shape[0]])
          .dataSync(),
      ];

      console.log(`[${i}] Loss: ${loss}`);

      self.postMessage({
        type: "optim-step:response",
        payload: {
          loss,
          gLoss,
          dLoss,
          fakeOutputsSync,
          ganRangeOutputSync,
          ganRangeDOutputSync,
          dFakeOutputsSync,
          dTargetDataSync,
        },
      });

      i++;
    }
  };
}

main(OneDGAN, data, tf);
