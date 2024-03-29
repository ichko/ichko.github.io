define(() => {
  const range = n => Array.from(Array(n).keys());

  function parallel(handler) {
    if (!window.Worker) {
      return handler;
    }

    // Idea - <https://github.com/zevero/worker-create>
    const handlerStr = handler.toString();
    const handlerBlob = new Blob([`
            'use strict';
            self.onmessage = async (e) => {
                const {args, callId} = e.data;
                try {
                    const handler = ${handlerStr}; 
                    const value = await handler.apply(null, args);
                    postMessage({value, callId, isError: false});
                } catch (e) {
                    postMessage({value: e, callId, isError: true});
                }
            };
        `], {
      type: 'text/javascript'
    });

    const url = window.URL.createObjectURL(handlerBlob);
    const worker = new window.Worker(url);

    let callId = 0;
    const calls = {};

    worker.onmessage = ({
      data: {
        callId,
        value,
        isError
      }
    }) => {
      if (calls[callId]) {
        if (isError) {
          calls[callId].reject(value);
        } else {
          calls[callId].resolve(value);
        }
        delete calls[callId];
      } else {
        throw new Error(`Missing resolver for callId: ${callId}`);
      }
    };

    worker.onerror = e => {
      throw new Error(`This should not happen, but it did with error: ${e}`);
    };

    return function () {
      callId++;
      let args = [...arguments];

      return new Promise((resolve, reject) => {
        calls[callId] = {
          resolve,
          reject
        };
        worker.postMessage({
          args,
          callId
        });
      })
    };
  }

  async function wait(duration) {
    return new Promise((resolve, _reject) => {
      setTimeout(() => resolve(), duration);
    });
  }

  // Works with SVG diagrams generated with <draw.io>
  function prepareDiagramElement({
    svgId,
    textContent
  }) {
    // const svg = document.getElementById(svgId).contentDocument;
    const svg = document.getElementById(svgId);

    // This comes from the way draw.io rendered to svg
    const divs = Array.from(svg.querySelectorAll('div'))
      .filter(el => el.textContent === textContent);
    const contentSelected = divs[divs.length - 1];
    const idSelected = document.getElementById(textContent);
    const div = contentSelected || idSelected;

    const rectSibling = div
      .parentElement.parentElement.parentElement
      .parentElement.parentElement.previousElementSibling;

    // const {width, height} = rectSibling.getBoundingClientRect();
    const {
      width,
      height
    } = rectSibling.getBBox();
    const [W, H] = [width, height].map(Math.round).map(a => a - 6);

    div.style.display = 'block';
    div.style.width = `${W}px`;
    div.style.height = `${H}px`;
    div.style.overflow = 'hidden';
    div.id = textContent;
    div.innerText = '';

    return [div, {
      width: W,
      height: H
    }];
  }

  function monkeyPatchSVGContext(svgId, handler) {
    const svgWin = document.getElementById(svgId).contentWindow;
    const oldElementCtor = window.HTMLElement;
    window.HTMLElement = svgWin.HTMLElement;
    handler();
    window.HTMLElement = oldElementCtor;
  }

  // SRC - <https://stackoverflow.com/a/14070928>
  async function loadSVGs({
    selector,
    dataAttr = 'data-url'
  } = {}) {
    const promises = Array.from(document.querySelectorAll(selector))
      .map(async e => {
        const svgUrl = e.attributes[dataAttr].value;
        const content = await fetch(svgUrl);
        e.innerHTML = await content.text();
      });

    await Promise.all(promises);
  }


  const histogram = parallel(({
    min,
    max,
    data,
    normalized,
    numBins,
    binSize
  } = {
    normalized: false
  }) => {
    // This has to be here because this is worker function.
    const range = n => Array.from(Array(n).keys());

    min = min !== undefined ? min : data.reduce((a, b) => Math.min(a, b), data[0]);
    max = max !== undefined ? max : data.reduce((a, b) => Math.max(a, b), data[0]);

    if (binSize === undefined) {
      binSize = (max - min) / numBins;
    } else if (numBins == undefined) {
      numBins = Math.ceil((max - min) / binSize);
    }

    const x = range(numBins).map(i => (i * binSize) + binSize / 2 + min);
    let y = range(numBins).map(() => 0);
    data.forEach(el => {
      const id = Math.floor((el - min) / (max - min) * (numBins - 1));
      y[id]++;
    });

    if (normalized) {
      y = y.map(y => y / data.length);
    }

    return {
      x,
      y,
      binSize,
      numBins,
      min,
      max
    };
  });

  return {
    range,
    parallel,
    prepareDiagramElement,
    monkeyPatchSVGContext,
    loadSVGs,
    histogram,
    wait,
  };
});
