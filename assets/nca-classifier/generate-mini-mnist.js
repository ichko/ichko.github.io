import mnist from "https://esm.sh/mnist@1.1.0";

export function generateMiniMNIST() {
    const result = {};
    console.log(mnist)
    mnist.forEach((obj, id) => {
        result[id] = []
        for (let i = 0; i < 10; i++) {
            const image = obj.get(i);
            result[id].push(image);
        }
    });

    const data = JSON.stringify(result);
    const blob = new Blob([data], { type: 'text/plain' });
    const fileURL = URL.createObjectURL(blob);

    const downloadLink = document.createElement('a');
    downloadLink.href = fileURL;
    downloadLink.download = 'mini-mnist.json';
    document.body.appendChild(downloadLink);
    downloadLink.click();
}
