const range = n => Array.from(Array(n).keys());

// const hex = ["eac435","345995","e40066","03cea4","fb4d3d","57b8ff","4c230a","280004"];
// const hex = ["08415c","cc2936","ebbab9","388697","b5ffe1","432534","412234","08a4bd"];
const hex = ["fc0","3d9","3df","d66","66c","f60","fc0"];

const C = hex.map(c => `#${c}`);
const colors = {
    z: C[0],
    g: C[1],
    gz: C[6],
    x: C[2],
    d: C[3],
    dgz: C[4],
    dx: C[5],
    dLoss: C[3],
    gLoss: C[1],
};

const generateData = data.bimodal(0.3);

const hideGridAx = {
    showgrid: false,
    showline: false,
    zeroline: false,
    showticklabels: false,
};

function initInputDataUI(fixedInputSync) {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'first-box',
    });

    const trace = {
        name: 'Input data',
        x: fixedInputSync,
        type: 'histogram',
        opacity: 0.8,
        marker: { color: colors.z },
        histnorm: 'probability',
        xbins: { size: 0.3 },
        legendgroup: 'distrib',
    };

    const layout = {
        margin: { r: 1, t: 20, b: 25, l: 1 },
        dragmode: 'pan',
        showlegend: false,
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        xaxis: { showgrid: false },
        yaxis: { showgrid: false },
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [trace], layout, {
        displayModeBar: false,
        staticPlot: true,
        responsive: true,
    });
};

function initGANViewUI(x, y) {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'second-box',
    });

    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        line: { width: 5, color: colors.g }
    };

    const layout = {
        margin: { r: 1, t: 1, b: 1, l: 1 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [trace], layout, {
        displayModeBar: false,
        staticPlot: true,
        responsive: true,
    });

    return ySynced => {
        Plotly.restyle(containerEl, { y: ySynced });
    };
}

function initGANOutputUI(output, target) {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'third-box',
    });

    const targetTrace = {
        x: target,
        marker: { color: colors.x },
        type: 'histogram',
        opacity: 0.8,
        histnorm: 'probability',
        xbins: { size: 0.4 },
        name: '$X$',
    };

    const outputTrace = {
        marker: { color: colors.gz },
        x: output,
        type: 'histogram',
        histnorm: 'probability',
        opacity: 0.9,
        xbins: { size: 0.4 },
        name: '$G(z)$',
    };

    const ySampleSize = 50;
    const ySamples = range(ySampleSize).map(() => 0)
    const tickersTrace = {
        marker: { color: colors.gz, opacity: 0.8, symbol: 'line-ns-open', },
        mode: 'markers',
        type: 'scatter',
        x: ySamples,
        y: ySamples,
        yaxis: 'y2',
        showlegend: false,
    };

    const layout = {
        margin: { r: 1, t: 5, b: 5, l: 1 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        xaxis: { range: [-3.5, 3.5] },
        yaxis: { range: [0, 0.32], domain: [0.15, 1] },
        yaxis2: {
            range: [-3.5, 3.5],
            domain: [0, 0.1],
            showgrid: false, 
            zerolinecolor: '#ccc',
        },
        barmode: 'overlay',
        showlegend: true,
        legend: {
          x: 0,
          xanchor: 'left',
          y: 0.9
        },
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [targetTrace, outputTrace, tickersTrace], layout, {
        displayModeBar: false,
        staticPlot: true,
    });

    return fakeOutputsSync => {
        Plotly.restyle(containerEl, { x: fakeOutputsSync }, 1);
        Plotly.restyle(containerEl, { x: fakeOutputsSync.slice(0, ySampleSize) }, 2);
    };
}

function initLossUI() {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'loss-box',
    });

    const dLossTrace = {
        y: [0.5, 1],
        type: 'scatter',
        mode: 'lines',
        opacity: 0.8,
        line: { width: 3, color: colors.dLoss },
        name: '$D\ Loss$',
    };

    const gLossTrace = {
        y: [0.5, 1],
        type: 'scatter',
        mode: 'lines',
        opacity: 0.8,
        line: { width: 3, color: colors.gLoss },
        name: '$G\ Loss$',
    };

    const layout = {
        margin: { r: 0, t: 0, b: 0, l: 0 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        xaxis1: { ...hideGridAx },
        yaxis1: { ...hideGridAx },
        showlegend: true,
        legend: {
          x: 0,
          y: 1,
          xanchor: 'left',
          bgcolor: 'rgba(255,255,255,0.4)',
        },
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [dLossTrace, gLossTrace], layout, {
        displayModeBar: false,
        staticPlot: true,
    });

    return (dLoss, gLoss) => {
        Plotly.extendTraces(containerEl, { y: [[dLoss], [gLoss]] }, [0, 1], 50);
        // monkeyPatchSVGContext('svg-object', () => {});
    };
}

function initDInputBoxesUI(output, target) {
    const [gzContainerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'gz-bottom-box',
    });
    const [xContainerEl, _layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'x-bottom-box',
    });

    const targetTrace = {
        x: target,
        marker: { color: colors.x },
        type: 'histogram',
        histnorm: 'probability',
        xbins: { size: 0.2 },
    };

    const outputTrace = {
        x: output,
        marker: { color: colors.gz },
        type: 'histogram',
        histnorm: 'probability',
        xbins: { size: 0.2 },
    };

    const layout = {
        margin: { r: 1, t: 1, b: 1, l: 1 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        xaxis: { range: [-3.5, 3.5] },
        yaxis: { range: [0, 0.13], domain: [0.15, 1] },
        showlegend: false,
        ...layoutSize,
    };

    Plotly.purge(gzContainerEl);
    Plotly.purge(xContainerEl);

    Plotly.newPlot(gzContainerEl, [outputTrace], layout, {
        displayModeBar: false,
        staticPlot: true,
    });

    Plotly.newPlot(xContainerEl, [targetTrace], layout, {
        displayModeBar: false,
        staticPlot: true,
    });

    return fakeOutputsSync => {
        Plotly.restyle(gzContainerEl, { x: fakeOutputsSync }, 0);
    };
}

function initDBoxUI(x, y) {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'd-box',
    });

    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        line: { width: 5, color: colors.d }
    };

    const layout = {
        margin: { r: 1, t: 1, b: 1, l: 1 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [trace], layout, {
        displayModeBar: false,
        staticPlot: true,
        responsive: true,
    });

    return ySynced => {
        Plotly.restyle(containerEl, { y: ySynced }, 0);
    };
}

function initDOutputUI(dx, dgz) {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'dgz-box',
    });

    const dxTrace = {
        x: dx,
        marker: { color: colors.dx },
        type: 'histogram',
        opacity: 0.8,
        histnorm: 'probability',
        xbins: { size: 0.05 },
        name: '$D(X)$',
    };

    const dgzTrace = {
        marker: { color: colors.dgz },
        x: dgz,
        type: 'histogram',
        histnorm: 'probability',
        opacity: 0.8,
        xbins: { size: 0.05 },
        name: '$D(G(z))$',
    };

    const layout = {
        margin: { r: 1, t: 5, b: 5, l: 1 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        xaxis: { range: [0.01, 0.99] },
        yaxis: { range: [0, 0.5], domain: [0.08, 1] },
        barmode: 'overlay',
        showlegend: true,
        legend: {
          x: 0,
          y: 0.87,
          xanchor: 'left',
        },
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [dgzTrace, dxTrace], layout, {
        displayModeBar: false,
        staticPlot: true,
    });

    return (dx, dgz) => {
        Plotly.restyle(containerEl, { x: dgz }, 0);
        Plotly.restyle(containerEl, { x: dx }, 1);
    };
}

document.body.onload = async () => {
    let t = 0;
    let running = false;

    await loadSVGs({selector: '.svg'});

    let gan = undefined;

    const init = () => {
        const inpDims = 1;
        gan = new OneDGAN(0.005, inpDims);

        const targetData = generateData(512);
        const targetDataSync = targetData.dataSync();
        const fixedInputs = gan.sampleZ([512, inpDims]);
        const fixedInputSync = fixedInputs.dataSync();
        const fakeOutputs = gan.G.predict(fixedInputs);
        const fakeOutputsSync = fakeOutputs.dataSync();

        const rangeInputs = tf.range(-3, 3, 0.1).expandDims(1);
        const rangeInputSync = rangeInputs.dataSync();

        initInputDataUI(fixedInputSync);
        const updateGANView = initGANViewUI(rangeInputSync, gan.G.predict(rangeInputs).dataSync());
        const updateDBox = initDBoxUI(rangeInputSync, gan.D.predict(rangeInputs).dataSync());
        const updateGANOutputs = initGANOutputUI(fakeOutputsSync, targetDataSync);
        const updateLoss = initLossUI();
        const updateDInputBox = initDInputBoxesUI(fakeOutputsSync, targetDataSync);
        const updateDOutputBox = initDOutputUI(
            gan.D.predict(fakeOutputs).dataSync(),
            gan.D.predict(targetData).dataSync(),
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
            init();
        };

        // This is done for optimization (single pass of the generator)
        const combinedGInput = fixedInputs.concat(rangeInputs);
        const combinedDInput = rangeInputs.concat(fakeOutputs).concat(targetData);
        
        async function step(i) {
            const batch = generateData(128);
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

            if (i % 10 == 0) {
                updateGANOutputs(fakeOutputsSync);
                updateDInputBox(fakeOutputsSync);
                updateDOutputBox(dFakeOutputsSync, dTargetDataSync);
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
    };

    init();
};
