const range = n => Array.from(Array(n).keys());

function generateData(size) {
    const frac = 0.3;

    const leftSize = Math.floor(size * frac);
    const rightSize = Math.floor(size * (1 - frac));

    const firstMode = tf.randomNormal([leftSize, 1], -1.5, 0.5);
    const secondMode = tf.randomNormal([rightSize, 1], 1.5, 0.5);
    const bimodalData = firstMode.concat(secondMode);

    tf.util.shuffle(bimodalData);

    return bimodalData;
}

let gan, fixedInputs;
const ySampleSize = 100;
let t = 0;
let rangeInputs;

function init() {
    t = 0;

    const inpDims = 1;
    const bimodalData = generateData(5000).dataSync();
    fixedInputs = tf.randomNormal([5000, inpDims]);
    const fixedInputSync = fixedInputs.dataSync();
    gan = new OneDGAN(0.008, inpDims);
    const fakeOutputs = gan.G.predict(fixedInputs).dataSync();

    rangeInputs = tf.range(-2, 2, 0.1);
    const rangeInputSync = rangeInputs.dataSync();

    const inputDataTrace = {
        name: 'Input data',
        x: fixedInputSync,
        type: 'histogram',
        opacity: 0.8,
        histnorm: 'probability',
        xbins: { size: 0.2 },
        xaxis: 'x4',
        yaxis: 'y4',
        legendgroup: 'distrib',
    }

    const trainingDataTrace = {
        name: 'Target data',
        x: bimodalData,
        type: 'histogram',
        opacity: 0.3,
        histnorm: 'probability',
        xbins: { size: 0.2 },
        xaxis: 'x1',
        yaxis: 'y1',
        legendgroup: 'distrib',
    };

    const generatedDataTrace = {
        name: 'GAN data',
        x: fakeOutputs,
        type: 'histogram',
        histnorm: 'probability',
        opacity: 0.8,
        xbins: { size: 0.2 },
        xaxis: 'x1',
        yaxis: 'y1',
        legendgroup: 'distrib',
    };


    const ySamples = range(ySampleSize).map(() => 0)
    const tickersTrace = {
        marker: { color: 'orange', symbol: 'line-ns-open', opacity: 0.8 },
        mode: 'markers',
        showlegend: false,
        type: 'scatter',
        x: ySamples,
        y: ySamples,
        xaxis: 'x1',
        yaxis: 'y2',
    };

    const dLossTrace = {
        y: [0, 1],
        type: 'scatter',
        mode: 'lines',
        xaxis: 'x3',
        yaxis: 'y3',
        name: 'D Loss',
        legendgroup: 'loss',
    };

    const gLossTrace = {
        y: [0, 1],
        type: 'scatter',
        mode: 'lines',
        xaxis: 'x3',
        yaxis: 'y3',
        name: 'G Loss',
        legendgroup: 'loss',
    };

    const generatorFunctionTrace = {
        x: rangeInputSync,
        y: gan.G.predict(rangeInputs).dataSync(),
        type: 'scatter',
        mode: 'lines',
        xaxis: 'x6',
        yaxis: 'y6',
        name: 'G Landscape',
        legendgroup: 'loss',
    };

    const hideGridAx = {
        showgrid: false,
        showline: false,
        zeroline: false,
        showticklabels: false
    };

    const data = [
        trainingDataTrace,
        generatedDataTrace,
        tickersTrace,
        gLossTrace,
        dLossTrace,
        inputDataTrace,
        generatorFunctionTrace
    ];

    const layout = {
        margin: {
            r: 10,
            t: 30,
            b: 10,
            l: 10
        },
        // annotations: [
        //     {
        //         x: 0.5,
        //         y: 0.5,
        //         xref: 'x5',
        //         yref: 'y5',
        //         text: '$G(x)$',
        //         arrowhead: 4,
        //         ax: -50,
        //         ay: -20
        //     }
        // ],
        barmode: 'overlay',
        dragmode: 'pan',
        height: 350,
        width: 750,
        legend: { orientation: 'h', y: 1 },

        xaxis4: { range: [-3.5, 3.5], domain: [0, 0.15] },
        yaxis4: { range: [0, 0.13], domain: [0.35, 0.65] },

        xaxis6: { domain: [0.2, 0.45] },
        yaxis6: { domain: [0.30, 0.70] },

        xaxis1: { range: [-3.5, 3.5], domain: [0.55, 1] },
        yaxis1: { range: [0, 0.13], domain: [0.15, 0.8] },

        xaxis5: { range: [0, 1], domain: [0, 1], ...hideGridAx },
        yaxis5: { range: [0, 1], domain: [0, 1], ...hideGridAx },

        xaxis3: { domain: [0.85, 1] },
        yaxis3: { domain: [0.85, 1.0] },

        yaxis2: {
            domain: [0, 0.1],
            ...hideGridAx,
            zeroline: true,
            zerolinecolor: '#ccc',
        },
        grid: { rows: 3, columns: 1, pattern: 'independent' },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)'
    };

    Plotly.purge('gan-output');

    Plotly.newPlot('gan-output', data, layout, {
        displayModeBar: false,
        staticPlot: true
    });
}

document.body.onload = async () => {
    init();

    let running = false;
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
        const batch = generateData(16);
        const loss = await gan.optim_step(batch);
        const [gLoss, dLoss] = loss;

        console.log(`[${i}] Loss: ${loss}`);

        const fakeOutputs = gan.G.predict(fixedInputs).dataSync();


        Plotly.update('gan-output', { x: fakeOutputs }, {}, 1);
        Plotly.update('gan-output', { y: gan.G.predict(rangeInputs).dataSync() }, {}, 6);
        Plotly.update('gan-output', { x: fakeOutputs.slice(0, ySampleSize) }, {}, 2);
        Plotly.extendTraces('gan-output', { y: [[dLoss], [gLoss]] }, [3, 4], 100);

        await tf.nextFrame();
    }

    const loop = async () => {
        if (running) {
            t += 1;
            await step(t);
            itInfoEl.innerText = '#' + t.toString().padStart(4, 0);
        }

        window.requestAnimationFrame(loop);
    };

    loop();
};
