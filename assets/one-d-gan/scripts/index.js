requirejs.config({
    paths: {
        tf: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf',
        tfvis: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.0.2/dist/tfjs-vis.umd.min',
        plotly: 'https://cdn.plot.ly/plotly-latest.min',

        gan: '/assets/one-d-gan/scripts/gan',
        utils: '/assets/one-d-gan/scripts/utils',
        ui: '/assets/one-d-gan/scripts/ui',
    }
});

requirejs(
    ['tf', 'gan', 'ui', 'utils'],
    () => main()
);

async function main() {
    const tf = require('tf');
    const {
        data,
        OneDGAN
    } = require('gan');
    const ui = require('ui');
    const {
        loadSVGs
    } = require('utils');

    const generateData = data.bimodal(0.3);

    await loadSVGs({
        selector: '.svg'
    });

    async function init(running, firstCall) {
        let t = 0;
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

        ui.initInputDataUI(fixedInputSync);
        const updateGANView = ui.initGANViewUI(rangeInputSync, gan.G.predict(rangeInputs).dataSync());
        const updateDBox = ui.initDBoxUI(rangeInputSync, gan.D.predict(rangeInputs).dataSync());
        const updateLoss = ui.initLossUI();

        const updateGANOutputs = await ui.initGANOutputUI(fakeOutputsSync, targetDataSync);
        // const updateDInputBox = await ui.initDInputBoxesUI(fakeOutputsSync, targetDataSync);
        const updateDOutputBox = await ui.initDOutputUI(
            gan.D.predict(targetData).dataSync(),
            gan.D.predict(fakeOutputs).dataSync(),
        );

        const playPauseEl = document.getElementById('play-pause');
        const itInfoEl = document.getElementById('iteration-info');
        const resetEl = document.getElementById('reset');

        function setButtonLabel() {
            const label = running ? 'Pause' : 'Play';
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

        // This is done for optimization (single pass of the generator)
        const combinedGInput = fixedInputs.concat(rangeInputs);
        const combinedDInput = rangeInputs.concat(fakeOutputs).concat(targetData);

        async function step(i) {
            const bs = i == 0 ? 2 : 128;
            const batch = generateData(bs);
            const loss = await gan.optim_step(batch);
            const [gLoss, dLoss] = loss;

            console.log(`[${i}] Loss: ${loss}`);

            const combinedGOutput = gan.G.predict(combinedGInput);
            const [fakeOutputsSync, ganRangeOutputSync] = [
                combinedGOutput.slice([0], [fixedInputs.shape[0]]).dataSync(),
                combinedGOutput.slice([fixedInputs.shape[0]], [rangeInputs.shape[0]]).dataSync(),
            ];

            const combinedDOutput = gan.D.predict(combinedDInput);
            const [ganRangeDOutputSync, dFakeOutputsSync, dTargetDataSync] = [
                combinedDOutput.slice([0], [rangeInputs.shape[0]]).dataSync(),
                combinedDOutput.slice([rangeInputs.shape[0]], [fakeOutputs.shape[0]]).dataSync(),
                combinedDOutput.slice([fakeOutputs.shape[0]], [targetData.shape[0]]).dataSync(),
            ];

            if (i % 5 == 0) {
                updateGANOutputs(fakeOutputsSync);
                // updateDInputBox(fakeOutputsSync);
                updateDOutputBox(dTargetDataSync, dFakeOutputsSync);
            }

            updateGANView(ganRangeOutputSync);
            updateDBox(ganRangeDOutputSync);
            updateLoss(dLoss, gLoss);
        }

        async function loop() {
            if (running) {
                t += 1;
                await step(t);
                itInfoEl.innerText = '#' + t.toString().padStart(4, 0);
            }

            window.requestAnimationFrame(loop);
        };

        loop();

        if (firstCall) {
            await step(0);
        }
    }

    await init(false, true);
    resolveLoading('main-demo');
}
