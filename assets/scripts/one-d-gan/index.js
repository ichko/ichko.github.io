const range = n => Array.from(Array(n).keys());
const SOFT_ONE = 0.95;

class OneDGAN {
    constructor(lr, inpDims, outDims = 1) {
        this.inpDims = inpDims;
        this.outDims = outDims;

        const G = (() => {
            const input = tf.input({ shape: [this.inpDims] });

            let x = input;
            x = tf.layers.dense({ units: 10, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 10, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: this.outDims, activation: 'linear' }).apply(x);
            return tf.model({ inputs: input, outputs: x });
        })();

        const D = (() => {
            const input = tf.input({ shape: [this.outDims] });

            let x = input;
            x = tf.layers.dense({ units: 10, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 10, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x);
            return tf.model({ inputs: input, outputs: x });
        })();

        this.G = G;
        this.D = D;

        this.D.compile({
            optimizer: tf.train.adam(lr, 0.5),
            loss: ['binaryCrossentropy']
        });

        this.combined = (() => {
            let input = tf.input({ shape: [inpDims] });
            let x = input;
            x = this.G.apply(input);
            this.D.trainable = false;
            x = this.D.apply(x);
            return tf.model({ inputs: input, outputs: x });
        })();

        this.combined.compile({
            optimizer: tf.train.adam(lr, 0.5),
            loss: ['binaryCrossentropy']
        });

    }

    async optim_step(X_real) {
        const [bs] = X_real.shape;
        // this.D.trainable = true;
        const loss_d = await this.optim_step_D(X_real);
        // this.D.trainable = false;
        const loss_g = await this.optim_step_G(bs * 2);

        return [loss_g, loss_d];
    }

    async optim_step_D(X_real) {
        const [bs] = X_real.shape;

        const [X, y] = tf.tidy(() => {
            const noise = tf.randomNormal([bs, this.inpDims]);
            const X_fake = this.G.predict(noise);

            const X = tf.concat([X_real, X_fake], 0);

            const y = tf.tidy(
                () => tf.concat([tf.ones([bs, 1]).mul(SOFT_ONE), tf.zeros([bs, 1])])
            );

            return [X, y];
        });

        const loss = await this.D.trainOnBatch(X, y);
        tf.dispose([X, y]);

        return loss;
    }

    async optim_step_G(bs) {
        const [X, y] = tf.tidy(() => {
            const noise = tf.randomNormal([bs, this.inpDims]);
            const y = tf.ones([bs, 1]).mul(SOFT_ONE);

            return [noise, y];
        });

        const loss = await this.combined.trainOnBatch(X, y);
        tf.dispose([X, y]);

        return loss;
    }
}

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

document.body.onload = async() => {
    const inpDims = 3;
    const bimodalData = generateData(5000).dataSync();
    const fixedInputs = tf.randomNormal([5000, inpDims]);
    const fixedInputSync = fixedInputs.dataSync();

    const gan = new OneDGAN(0.002, inpDims);

    var trace = {
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

    var trace2 = {
        name: 'GAN data',
        x: fixedInputSync,
        type: 'histogram',
        histnorm: 'probability',
        opacity: 0.8,
        xbins: { size: 0.2 },
        xaxis: 'x1',
        yaxis: 'y1',
        legendgroup: 'distrib',
    };


    const ySampleSize = 100;
    const ySamples = range(ySampleSize).map(() => 0)
    const tickersTrace = {
        marker: { color: 'orange', symbol: 'line-ns-open' },
        mode: 'markers',
        name: 'Group 2',
        showlegend: false,
        type: 'scatter',
        x: ySamples,
        y: ySamples,
        xaxis: 'x1',
        yaxis: 'y2',
    };

    var dLossTrace = {
        y: [],
        type: 'scatter',
        mode: 'lines',
        xaxis: 'x3',
        yaxis: 'y3',
        name: 'D Loss',
        legendgroup: 'loss',
    };

    var gLossTrace = {
        y: [],
        type: 'scatter',
        mode: 'lines',
        xaxis: 'x3',
        yaxis: 'y3',
        name: 'G Loss',
        legendgroup: 'loss',
    };

    var data = [trace, trace2, tickersTrace, gLossTrace, dLossTrace];
    var layout = {
        barmode: 'overlay',
        dragmode: 'pan',
        height: 700,
        legend: { x: 1, y: 1 },
        xaxis1: {
            range: [-3.5, 3.5],
            domain: [0, 1],
            ticks: 'outside',
        },
        xaxis3: { ticks: 'outside' },
        yaxis1: { range: [0, 0.13], domain: [0.44, 1] },
        yaxis2: {
            domain: [0.35, 0.4],
            showgrid: false,
            showline: false,
            zeroline: true,
            zerolinecolor: '#ccc',
            showticklabels: false
        },
        yaxis3: { domain: [0, 0.3] },
        grid: { rows: 3, columns: 1, pattern: 'independent' },
    };
    Plotly.newPlot('myDiv', data, layout, {
        displayModeBar: false,
        staticPlot: true
    });

    const losses = [
        [],
        []
    ];

    async function step(i) {
        const batch = generateData(512);
        const loss = await gan.optim_step(batch);
        const [gLoss, dLoss] = loss;
        losses[0].push(dLoss);
        losses[1].push(gLoss);

        console.log(`[${i}] Loss: ${loss}`);

        const fakeOutputs = gan.G.predict(fixedInputs).dataSync();

        Plotly.update('myDiv', { x: fakeOutputs }, {}, 1);
        Plotly.update('myDiv', { x: fakeOutputs.slice(0, ySampleSize) }, {}, 2);
        Plotly.extendTraces('myDiv', {
            y: [
                [dLoss],
                [gLoss],
            ]
        }, [3, 4], 500)

        // await tf.nextFrame();
    }

    let t = 0;
    const loop = async() => {
        t += 1;
        for (let i = 0; i < 10; i++) {
            await step(t);
        }

        if (t < 1000) {
            window.requestAnimationFrame(loop);
        }
    };

    loop();
};