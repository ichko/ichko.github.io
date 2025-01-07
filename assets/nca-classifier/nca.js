function circularPad(x, padWidth) {
    const [top, bottom, left, right] = padWidth;

    let paddedX = tf.concat([
        x.slice([0, x.shape[1] - top, 0, 0], [-1, -1, -1, -1]),
        x,
        x.slice([0, 0, 0, 0], [-1, bottom, -1, -1])
    ], 1);

    paddedX = tf.concat([
        paddedX.slice([0, 0, paddedX.shape[2] - left, 0], [-1, -1, -1, -1]),
        paddedX,
        paddedX.slice([0, 0, 0, 0], [-1, -1, right, -1])
    ], 2);

    return paddedX;
}

export default class NCA extends tf.layers.Layer {
    static get className() {
      return 'NCA';
    }

    constructor() {
        super({})
        this.sobelX = tf.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
            .div(8).expandDims(-1).expandDims(-1);
        this.sobelY = tf.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
            .div(8).expandDims(-1).expandDims(-1);
        this.identity = tf.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
            .expandDims(-1).expandDims(-1);

        this.rule1 = tf.layers.conv2d({
            filters: 128,
            kernelSize: [1, 1],
            padding: "same",
        });
        this.rule2 = tf.layers.conv2d({
            filters: 16,
            kernelSize: [1, 1],
            padding: "same",
            useBias: false,
        });
    }

    call(x) {
        const oldX = x;
        const [BS, H, W, C] = x.shape;
        x = x.transpose([0, 3, 1, 2]).reshape([C * BS, H, W, 1]);
        x = circularPad(x, [1, 1, 1, 1]);
        const x1 = oldX
        const x2 = tf.conv2d(x, this.sobelX, 1, 0).reshape([BS, C, H, W]).transpose([0, 2, 3, 1]).reshape([BS, H, W, C]);
        const x3 = tf.conv2d(x, this.sobelY, 1, 0).reshape([BS, C, H, W]).transpose([0, 2, 3, 1]).reshape([BS, H, W, C]);
        // console.log(x1.shape)
        x = tf.stack([x1, x2, x3], -1).reshape([BS, H, W, -1]);
        x = this.rule1.apply(x);
        x = tf.relu(x);
        x = this.rule2.apply(x);
        x = oldX.add(x);
        return x;
    }
}

export async function loadWeights(nca) {
    const weightsURL = 
        "https://gist.githubusercontent.com/ichko/" +
        "90325bb0970dadc66f0fd27c33d9077c/raw/" +
        "d3c2726ce46da3c0b028500f1f905eda2f029020/gistfile1.txt";

    const response = await fetch(weightsURL);
    const data = await response.json();
    const tensorData = {};
    console.log(data);
    for (const [key, value] of Object.entries(data)) {
        if (key.startsWith("kernel") || key.startsWith("seed")) continue;

        const tensor =  tf.tensor(value)
        tensorData[key] = tensor;
        if (tensor.shape.length == 4) {
            tensorData[key] = tensorData[key].transpose([2, 3, 1, 0])
        }
        console.log(`${key}: ${tensorData[key].shape}`);
    }

    const [w1, b1] = nca.rule1.getWeights();
    w1.assign(tensorData["rule.0.weight"]);
    b1.assign(tensorData["rule.0.bias"]);

    const [w2] = nca.rule2.getWeights();
    w2.assign(tensorData["rule.2.weight"]);
}
