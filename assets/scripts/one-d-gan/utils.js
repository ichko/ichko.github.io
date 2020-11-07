// Works with SVG diagrams generated with <draw.io>
function prepareDiagramElement({svgId, textContent}) {
    // const svg = document.getElementById(svgId).contentDocument;
    const svg = document.getElementById(svgId);

    // This comes from the way draw.io rendered to svg
    const divs = Array.from(svg.querySelectorAll('div'))
        .filter(el => el.textContent === textContent);
    const contentSelected = divs[divs.length - 1];
    const idSelected = document.getElementById(textContent);
    const div = contentSelected || idSelected;

    const rectSibling = div.parentElement.parentElement.parentElement.parentElement.parentElement.previousElementSibling;
    // const {width, height} = rectSibling.getBoundingClientRect();
    const {width, height} = rectSibling.getBBox();
    const [W, H] = [width, height].map(Math.round).map(a => a - 6);

    div.style.display = 'block';
    div.style.width = `${W}px`;
    div.style.height = `${H}px`;
    div.style.overflow = 'hidden';
    div.id = textContent;
    div.innerText = '';

    return [div, {width: W, height: H}];
}

function monkeyPatchSVGContext(svgId, handler) {
    const svgWin = document.getElementById(svgId).contentWindow;
    const oldElementCtor = window.HTMLElement;
    window.HTMLElement = svgWin.HTMLElement;
    handler();
    window.HTMLElement = oldElementCtor;
};

// SRC - <https://stackoverflow.com/a/14070928>
async function loadSVGs({selector, dataAttr='data-url'} = {}) {
    const promises = Array.from(document.querySelectorAll(selector)).map(async e => {
        const svgUrl = e.attributes[dataAttr].value;
        const content = await fetch(svgUrl);
        e.innerHTML = await content.text();
    });

    await Promise.all(promises);
}
