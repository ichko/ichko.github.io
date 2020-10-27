const range = n => Array.from(Array(n).keys());
const SOFT_ONE = 0.95;

class OneDGAN {
    constructor(lr) {
        const G = (() => {
            const input = tf.input({ shape: [1] });

            let x = input;
            x = tf.layers.dense({ units: 20, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 20, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 1, activation: 'linear' }).apply(x);
            return tf.model({ inputs: input, outputs: x });
        })();

        const D = (() => {
            const input = tf.input({ shape: [1] });

            let x = input;
            x = tf.layers.dense({ units: 20, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(x);
            return tf.model({ inputs: input, outputs: x });
        })();

        this.G = G;
        this.D = D;

        this.D.compile({
            optimizer: tf.train.sgd(lr, 0.5),
            loss: ['binaryCrossentropy']
        });

        this.combined = (() => {
            let input = tf.input({ shape: [1] });
            let x = input;
            x = this.G.apply(input);
            this.D.trainable = false;
            x = this.D.apply(x);
            return tf.model({ inputs: input, outputs: x });
        })();

        this.combined.compile({
            optimizer: tf.train.sgd(lr, 0.5),
            loss: ['binaryCrossentropy']
        });
    }

    configure_optim(lr) {
        this.dOptimizer = tf.train.sgd(lr);
        this.gOptimizer = tf.train.sgd(lr);
    }

    async optim_step(X_real) {
        const [bs] = X_real.shape;
        // this.D.trainable = true;
        const loss_d = await this.optim_step_D(X_real);
        // this.D.trainable = false;
        const loss_g = await this.optim_step_G(bs * 2);

        return [loss_d, loss_g];
    }

    async optim_step_D(X_real) {
        const [bs] = X_real.shape;

        const [X, y] = tf.tidy(() => {
            const noise = tf.randomNormal([bs, 1]);
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
            const noise = tf.randomNormal([bs, 1]);
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
    const bimodalData = generateData(5000);
    const fixedInputs = tf.randomNormal([5000, 1]);

    const gan = new OneDGAN(0.2);
    const its = 500;

    var trace = {
        name: 'bimodal',
        x: bimodalData.dataSync(),
        type: 'histogram',
        opacity: 0.8,
        xbins: {
            size: 0.1,
        }
    };
    var trace2 = {
        name: 'single mode',
        x: fixedInputs.dataSync(),
        type: 'histogram',
        opacity: 0.8,
        xbins: {
            size: 0.1,
        }
    };
    var data = [trace, trace2];
    var layout = {
        barmode: "overlay",
        showlegend: false,
        dragmode: 'pan',
        xaxis: { range: [-3.5, 3.5] }
    };
    Plotly.newPlot('myDiv', data, layout, {
        displayModeBar: false,
        staticPlot: true
    });

    async function step(i) {
        const batch = generateData(128);
        const loss = await gan.optim_step(batch);

        console.log(`[${i}] Loss: ${loss}`);

        const fakeOutputs = gan.G.predict(fixedInputs);

        Plotly.animate('myDiv', {
            data: [{ x: fakeOutputs.dataSync() }],
            traces: [1],
            // layout: {}
        }, {
            transition: {
                duration: 10,
                easing: 'cubic-in-out'
            },
            frame: {
                duration: 10
            }
        });

        await tf.nextFrame();
    }

    let i = 0;
    const loop = async() => {
        i += 1;
        await step(i);

        window.requestAnimationFrame(loop);
    };

    loop();
};