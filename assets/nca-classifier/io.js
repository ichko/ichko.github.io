export default class IO {
    constructor(element) {
        this.element = element;
        this.mouseDown = false;
        this.mousePos = [];
        this.element.addEventListener("mousedown", () => {
            this.mouseDown = true;
        });
        window.addEventListener("mouseup", () => {
            this.mouseDown = false;
        });
        this.element.addEventListener("mousemove", (e) => {
            this.mousePos = [e.offsetX, e.offsetY];
        });
    }
}
