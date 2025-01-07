import colormap from "https://esm.sh/colormap@2.3.2";

export function drawCircle(tensor, pos, radius, value = 1) {
    const [H, W, _] = tensor.shape;
    const screen = tf.stack(tf.meshgrid(tf.range(0, W), tf.range(0, H)), -1);

    const mask = screen
        .sub(pos)
        .norm(2, -1)
        .less(radius)
        .cast("float32")
        .expandDims(-1);
    // const fill = tf.fill([H, W, 1], value);
    const fill = tf.randomUniform([H, W, 1]);
    const maskTrue = mask.mul(fill);
    const maskFalse = mask.mul(-1).add(1).mul(tensor).slice([0, 0, 0], [-1, -1, 1]);
    const newFrame = maskTrue.add(maskFalse).tile([1, 1, 3]);
    const rest = tensor.slice([0, 0, 3], [-1, -1, -1]);
    const newFullFrame = tf.concat([newFrame, rest], -1);
    return newFullFrame;
}

export function circularPad(x, padWidth) {
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

export function tidyUpdate(dep, updateHandler) {
    const result = tf.tidy(updateHandler);
    if (result != dep) {
        dep.dispose();
    }
    return result;
}

export let cmap = colormap({
    colormap: 'hot',
    nshades: 256,
    format: 'rgb',
    alpha: 1
});

export function applyCmap(tensor, cmap) {
    cmap = tf.tensor(cmap);
    const min = tensor.min();
    const max = tensor.max();
    const norm = tensor.sub(min).div(max.sub(min));
    const intMapped = norm.mul(255).round().cast('int32');
    return cmap.gather(intMapped);
}
