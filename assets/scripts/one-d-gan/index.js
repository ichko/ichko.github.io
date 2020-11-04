const range = n => Array.from(Array(n).keys());

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
        marker: { color: '#FF7F0E' },
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
        line: { width: 5, color: '#cc0000' }
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
        Plotly.update(containerEl, { y: ySynced }, {}, 0);
    };
}

function initGANOutputUI(output, target) {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'third-box',
    });

    const targetTrace = {
        x: target,
        type: 'histogram',
        opacity: 0.5,
        histnorm: 'probability',
        xbins: { size: 0.2 },
    };

    const outputTrace = {
        x: output,
        type: 'histogram',
        histnorm: 'probability',
        opacity: 0.8,
        xbins: { size: 0.2 },
    };

    const ySampleSize = 50;
    const ySamples = range(ySampleSize).map(() => 0)
    const tickersTrace = {
        marker: { color: 'orange', symbol: 'line-ns-open', opacity: 0.8 },
        mode: 'markers',
        showlegend: false,
        type: 'scatter',
        x: ySamples,
        y: ySamples,
        yaxis: 'y2',
    };

    const layout = {
        margin: { r: 1, t: 70, b: 5, l: 1 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        xaxis: { range: [-3.5, 3.5] },
        yaxis: { range: [0, 0.13], domain: [0.15, 1] },
        yaxis2: {
            range: [-3.5, 3.5],
            domain: [0, 0.1],
            showgrid: false, 
            zerolinecolor: '#ccc',
        },
        barmode: 'overlay',
        showlegend: false,
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [targetTrace, outputTrace, tickersTrace], layout, {
        displayModeBar: false,
        staticPlot: true,
    });

    return fakeOutputsSync => {
        Plotly.update(containerEl, { x: fakeOutputsSync }, {}, 1);
        Plotly.update(containerEl, { x: fakeOutputsSync.slice(0, ySampleSize) }, {}, 2);
    };
}

function initLossUI() {
    const [containerEl, layoutSize] = prepareDiagramElement({
        svgId: 'svg-object',
        textContent: 'loss-box',
    });

    const dLossTrace = {
        y: [1, 1],
        type: 'scatter',
        mode: 'lines',
        line: { width: 3, color: '#66B2FF' }
    };

    const gLossTrace = {
        y: [1, 1],
        type: 'scatter',
        mode: 'lines',
        line: { width: 3, color: '#DB6E00' },
    };

    const layout = {
        margin: { r: 0, t: 0, b: 0, l: 0 },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        showlegend: false,
        xaxis1: { ...hideGridAx },
        yaxis1: { ...hideGridAx },
        ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [dLossTrace, gLossTrace], layout, {
        displayModeBar: false,
        staticPlot: true,
    });

    return (dLoss, gLoss) => {
        monkeyPatchSVGContext('svg-object', () => {
            Plotly.extendTraces(containerEl, { y: [[dLoss], [gLoss]] }, [0, 1], 50);
        });
    };
}

document.body.onload = async () => {
    let t = 0;
    let running = false;

    const init = () => {
        const inpDims = 1;
        const gan = new OneDGAN(0.005, inpDims);
        const bimodalDataSync = generateData(5000).dataSync();
        const fixedInputs = gan.sampleZ([5000, inpDims]);
        const fixedInputSync = fixedInputs.dataSync();
        const fakeOutputsSync = gan.G.predict(fixedInputs).dataSync();

        const rangeInputs = tf.range(-2, 2, 0.2);
        const rangeInputSync = rangeInputs.dataSync();

        initInputDataUI(fixedInputSync);
        const updateGANView = initGANViewUI(rangeInputSync, gan.G.predict(rangeInputs).dataSync());
        const updateGANOutputs = initGANOutputUI(fakeOutputsSync, bimodalDataSync);
        const updateLoss = initLossUI();

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


        async function step(i) {
            const batch = generateData(256);
            const loss = await gan.optim_step(batch);
            const [gLoss, dLoss] = loss;

            // console.log(`[${i}] Loss: ${loss}`);

            const fakeOutputsSync = gan.G.predict(fixedInputs).dataSync();
            const ganRangeOutputSynced = gan.G.predict(rangeInputs).dataSync()

            updateGANOutputs(fakeOutputsSync);
            updateGANView(ganRangeOutputSynced);
            updateLoss(dLoss, gLoss);

            await tf.nextFrame();
        }

        async function loop() {
            if (running) {
                t += 1;
                await step(t);
                itInfoEl.innerText = '#' + t.toString().padStart(4, 0);
            }
    
            window.requestAnimationFrame(loop);
        };

        return loop;
    }

    const loop = init();
    loop();
};
