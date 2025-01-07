import Stats from "https://esm.sh/stats.js";
import colormap from "https://esm.sh/colormap@2.3.2";

import NCA, {loadWeights} from "./nca.js";
import IO from "./io.js";

let cmap = colormap({
    colormap: 'viridis',
    nshades: 256,
    format: 'rgb',
    alpha: 1
});

function applyCmap(tensor, cmap) {
    cmap = tf.tensor(cmap)
    const min = tensor.min();
    const max = tensor.max();
    const norm = tensor.sub(min).div(max.sub(min));
    const intMapped =  norm.mul(255).round().cast('int32');
    return cmap.gather(intMapped);
}


function drawCircle(tensor, pos, radius, value) {
    const [H, W, _] = tensor.shape;
    const screen = tf.stack(tf.meshgrid(tf.range(0, W), tf.range(0, H)), -1);

    const mask = screen
        .sub(pos)
        .norm(2, -1)
        .less(radius)
        .cast("float32")
        .expandDims(-1);
    // const fill = tf.fill(tensor.shape, value);
    // const fill = tf.randomUniform(tensor.shape);
    const fill = tf.ones([H, W, 1]);
    const maskTrue = mask.mul(fill);
    const maskFalse = mask.mul(-1).add(1).mul(tensor).slice([0, 0, 0], [-1, -1, 1]);
    const newFrame = maskTrue.add(maskFalse).tile([1, 1, 3]);
    const rest = tensor.slice([0, 0, 3], [-1, -1, -1]);
    const newFullFrame = tf.concat([newFrame, rest], -1);
    return newFullFrame;
}


window.onload = async (e) => {
    await tf.setBackend("webgpu");

    const mnist = await (await fetch("/assets/nca-classifier/mini-mnist.json")).json();

    var stats = new Stats();
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    let nca = new NCA();
    let isRunning = false;

    const depth = 16;
    const W = 64;
    const H = 64;

    const canvas = document.getElementById("canvas");
    const canvasIO = new IO(canvas);
    canvas.width = W;
    canvas.height = H;

    document.getElementById("clear").onclick = () => {
        inp = tf.zeros([H, W, depth]);
    };
    document.getElementById("reset").onclick = () => {
        nca = new NCA();
    };
    document.getElementById("toggle").onclick = () => {
        isRunning = !isRunning;
    };
    document.getElementById("random").onclick = () => {
        inp = tf.tidy(() => {
            const randIndex = tf.randomUniformInt([1], 0, 10).dataSync()[0];
            const randIndex2 = tf.randomUniformInt([1], 0, 10).dataSync()[0];
            const p = (W - 28) / 2;
            const m = tf.tensor(mnist[randIndex][randIndex2], [28, 28])
                .pad([[p, p], [p, p]]).expandDims(-1).tile([1, 1, 3]);
            return tf.concat([m, tf.zeros([H, W, depth - 3])], -1);
        });
    };
    document.getElementById("step").onclick = () => {
        inp = tf.tidy(() => {
            return nca.call(inp.expandDims(0)).squeeze(0);
        });
    };

    let inp = tf.randomUniform([H, W, depth]);
    inp = tf.tidy(() => nca.call(inp.expandDims(0)).squeeze(0));
    await loadWeights(nca);

    let memBefore = tf.memory().numTensors;
    async function animate() {
        stats.begin();

        const s = 5;
        const newInp = tf.tidy(() => {
            let newInp = inp;
            if (canvasIO.mouseDown) {
                const [x, y] = canvasIO.mousePos;
                newInp = drawCircle(newInp, [x / s, y / s], 3, 0);
            }
            if (isRunning) {
                newInp = nca.call(newInp.expandDims(0)).squeeze(0);
            }

            let screen = newInp.slice([0, 0, 0], [-1, -1, 1]).squeeze(-1);
            screen = applyCmap(screen, cmap).slice([0, 0, 0], [-1, -1, 3]);
            screen = tf.image.resizeNearestNeighbor(screen, [W * s, H * s]);
            screen = screen.sub(screen.min()).div(screen.max().sub(screen.min()));
            tf.browser.draw(screen, canvas);

            return newInp;
        });
        if (newInp != inp) {
            inp.dispose();
            inp = newInp;
        }

        const memAfter = tf.memory().numTensors;
        if (memBefore < memAfter) {
            console.warn(`memory is increasing to ${memAfter} tensors`);
        }
        memBefore = memAfter;

        stats.end();
        window.requestAnimationFrame(animate);
    }
    window.requestAnimationFrame(animate);
}
