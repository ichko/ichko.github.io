requirejs.config({
  paths: {
    tf: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf",
    // tfvis: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.0.2/dist/tfjs-vis.umd.min',
    plotly: "https://cdn.plot.ly/plotly-1.58.2.min",

    gan: "/assets/one-d-gan/scripts/gan",
    utils: "/assets/one-d-gan/scripts/utils",
    ui: "/assets/one-d-gan/scripts/ui",
  },
});

requirejs(["tf", "gan", "ui", "utils"], () => (window.onload = () => main()));

async function main() {
  const ui = require("ui");
  let t = 0;
  const { loadSVGs } = require("utils");

  await loadSVGs({
    selector: ".svg",
  });
  var updateGANView, updateDBox, updateLoss, updateGANOutputs, updateDOutputBox;
  let predicting = false;

  const worker = new Worker("/assets/one-d-gan/scripts/worker.js");

  worker.onmessage = async ({ data }) => {
    if (data.type == "init") {
      const initData = data.payload;
      await init(false, true);
      resolveLoading("main-demo");

      ui.initInputDataUI(initData.fixedInputSync);
      updateGANView = ui.initGANViewUI(
        initData.rangeInputSync,
        initData.gPredictRangeInput
      );

      updateDBox = ui.initDBoxUI(
        initData.rangeInputSync,
        initData.dPredictRangeInput
      );
      updateLoss = ui.initLossUI();

      updateGANOutputs = await ui.initGANOutputUI(
        initData.fakeOutputsSync,
        initData.targetDataSync
      );
      updateDOutputBox = await ui.initDOutputUI(
        initData.dPredictTargetData,
        initData.dPredictFakeData
      );
    }

    if (data.type == "optim-step:response") {
      const optimData = data.payload;

      updateGANOutputs(optimData.fakeOutputsSync);
      updateDOutputBox(optimData.dTargetDataSync, optimData.dFakeOutputsSync);

      updateGANView(optimData.ganRangeOutputSync);
      updateDBox(optimData.ganRangeDOutputSync);
      updateLoss(optimData.dLoss, optimData.gLoss);

      predicting = false;
    }
  };

  async function init(running, firstCall) {
    t = 0;

    const playPauseEl = document.getElementById("play-pause");
    const itInfoEl = document.getElementById("iteration-info");
    const resetEl = document.getElementById("reset");

    function setButtonLabel() {
      const label = running ? "Pause" : "Play";
      playPauseEl.innerText = label;
    }
    setButtonLabel();

    playPauseEl.onclick = () => {
      running = !running;
      setButtonLabel();
    };

    resetEl.onclick = () => {
      init(running);
      running = false;
    };

    async function step() {
      if (!predicting) {
        predicting = true;
        worker.postMessage({ type: "optim-step:request" });
        itInfoEl.innerText = "#" + t.toString().padStart(4, 0);
      }
    }

    async function loop() {
      if (running) {
        t += 1;
        await step();
      }

      window.requestAnimationFrame(loop);
    }

    loop();
  }
}
