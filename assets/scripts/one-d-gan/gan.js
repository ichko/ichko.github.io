const SOFT_ONE = 0.95;

const data = {
    unimodal() {
        return size => tf.randomNormal([size, 1])
    },
    bimodal(frac = 0.3) {
        return size => {
            const leftSize = Math.floor(size * frac);
            const rightSize = Math.floor(size * (1 - frac));

            const firstMode = tf.randomNormal([leftSize, 1], -1.5, 0.5);
            const secondMode = tf.randomNormal([rightSize, 1], 1.5, 0.5);
            const bimodalData = firstMode.concat(secondMode);

            tf.util.shuffle(bimodalData);

            return bimodalData;
        }
    }
};

class OneDGAN {
    constructor(lr, inpDims, outDims = 1) {
        this.inpDims = inpDims;
        this.outDims = outDims;

        const G = (() => {
            const input = tf.input({ shape: [this.inpDims] });

            let x = input;
            x = tf.layers.dense({ units: 20, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 10, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: this.outDims, activation: 'linear' }).apply(x);
            return tf.model({ inputs: input, outputs: x });
        })();

        const D = (() => {
            const input = tf.input({ shape: [this.outDims] });

            let x = input;
            x = tf.layers.dense({ units: 10, activation: 'relu' }).apply(x);
            x = tf.layers.dense({ units: 20, activation: 'relu' }).apply(x);
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