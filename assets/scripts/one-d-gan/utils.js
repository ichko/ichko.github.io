// Works with SVG diagrams generated with <draw.io>
function prepareDiagramElement({svgId, textContent}) {
    const svg = document.getElementById(svgId).contentDocument

    // This comes from the way draw.io rendered to svg
    const divs = Array.from(svg.querySelectorAll('div'))
        .filter(el => el.textContent === textContent);
    const contentSelected = divs[divs.length - 1];
    const idSelected = svg.getElementById(textContent);
    const div = contentSelected || idSelected;

    const rectSibling = div.parentElement.parentElement.parentElement.parentElement.parentElement.previousElementSibling;

    const width = rectSibling.attributes.width.value;
    const height = rectSibling.attributes.height.value;

    div.style.display = 'block';
    div.style.width = `${width}px`;
    div.style.height = `${height}px`;
    div.style.overflow = 'hidden';
    div.id = textContent;
    div.innerText = '';

    return [div, {width, height}];
}

function monkeyPatchSVGContext(svgId, handler) {
    const svgWin = document.getElementById(svgId).contentWindow;
    const oldElementCtor = window.HTMLElement;
    window.HTMLElement = svgWin.HTMLElement;
    handler();
    window.HTMLElement = oldElementCtor;
};
