import Stats from "https://esm.sh/stats.js";

import { NCARunner } from "./nca.js";


window.onload = async (e) => {
    await tf.setBackend("webgpu");

    var stats = new Stats();
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    let isRunning = false;
    const depth = 16;
    const W = 64;
    const H = 64;

    const canvas = document.getElementById("canvas");
    canvas.width = W;
    canvas.height = H;

    const ncaRunner = new NCARunner(W, H, depth, canvas);

    document.getElementById("clear").onclick = () => {
        ncaRunner.clear();
    };
    document.getElementById("toggle").onclick = () => {
        isRunning = !isRunning;
    };
    document.getElementById("random").onclick = () => {
        ncaRunner.randomMNISTInit();
    };
    document.getElementById("step").onclick = () => {
        ncaRunner.step();
    };

    await ncaRunner.loadWeights();

    let memBefore = tf.memory().numTensors;
    async function animate() {
        stats.begin();

        if (isRunning) {
            ncaRunner.step();
        }
        ncaRunner.draw();

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
