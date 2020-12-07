define(require => {
  const {
    prepareDiagramElement,
    histogram,
    range
  } = require('utils');
  const Plotly = require('plotly');

  // const hex = ["eac435","345995","e40066","03cea4","fb4d3d","57b8ff","4c230a","280004"];
  // const hex = ["fc0","3d9","3df","d66","66c","f60","fc0"];
  const hex = ["ff7f0e", "82B366", "4a92c3", "B85450", "ff7f0e", "4a92c3", "ff7f0e"];
  // const hex = ['orange', undefined, undefined, undefined, undefined, undefined, 'orange'];

  const C = hex.map(c => `${c}`);
  const colors = {
    z: C[0],
    g: C[1],
    gz: C[6],
    x: C[2],
    d: C[3],
    dgz: C[4],
    dx: C[5],
    dLoss: C[3],
    gLoss: C[1],
  };

  const animDuration = 0;

  const defaultLayout = {
    template: 'plotly_white',
  };

  const hideGridAx = {
    showgrid: false,
    showline: false,
    zeroline: false,
    showticklabels: false,
  };

  async function initInputDataUI(fixedInputSync) {
    const [containerEl, layoutSize] = prepareDiagramElement({
      svgId: 'svg-object',
      textContent: 'first-box',
    });

    const trace = {
      name: 'Input data',
      x: fixedInputSync,
      type: 'histogram',
      opacity: 0.8,
      marker: {
        color: colors.z
      },
      histnorm: 'probability',
      xbins: {
        size: 0.3
      },
      legendgroup: 'distrib',
      ...defaultLayout,
    };

    const layout = {
      margin: {
        r: 1,
        t: 20,
        b: 25,
        l: 1
      },
      dragmode: 'pan',
      showlegend: false,
      plot_bgcolor: 'rgba(0, 0, 0, 0)',
      paper_bgcolor: 'rgba(0, 0, 0, 0)',
      xaxis: {
        showgrid: false
      },
      yaxis: {
        showgrid: false
      },
      ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [trace], layout, {
      displayModeBar: false,
      staticPlot: true,
      responsive: true,
    });
  };


  function initGANViewUI(x, y) {
    const [containerEl, layoutSize] = prepareDiagramElement({
      svgId: 'svg-object',
      textContent: 'second-box',
    });

    const trace = {
      x: x,
      y: y,
      type: 'scatter',
      mode: 'lines',
      line: {
        width: 5,
        color: colors.g
      }
    };

    const layout = {
      margin: {
        r: 1,
        t: 1,
        b: 20,
        l: 1
      },
      plot_bgcolor: 'rgba(0, 0, 0, 0)',
      paper_bgcolor: 'rgba(0, 0, 0, 0)',
      xaxis: {
        range: [-3, 3]
      },
      yaxis: {
        range: [-3, 3]
      },
      ...defaultLayout,
      ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [trace], layout, {
      displayModeBar: false,
      staticPlot: true,
      responsive: true,
    });

    return ySynced => {
      Plotly.restyle(containerEl, {
        y: ySynced
      });
    };
  }

  async function initGANOutputUI(output, target) {
    const [containerEl, layoutSize] = prepareDiagramElement({
      svgId: 'svg-object',
      textContent: 'third-box',
    });

    const {
      x: targetX,
      y: targetY,
    } = await histogram({
      data: target,
      min: -3.5,
      max: 3.5,
      numBins: 20,
      normalized: true
    });

    const {
      x: outputX,
      y: outputY
    } = await histogram({
      min: -3.5,
      max: 3.5,
      numBins: 20,
      data: output,
      normalized: true,
    });

    const targetTrace = {
      x: targetX,
      y: targetY,
      marker: {
        color: colors.x
      },
      type: 'bar',
      opacity: 0.8,
      name: '$X$',
    };

    const outputTrace = {
      x: outputX,
      y: outputY,
      marker: {
        color: colors.gz
      },
      type: 'bar',
      opacity: 0.8,
      name: '$G(z)$',
    };

    const ySampleSize = 50;
    const ySamples = range(ySampleSize).map(() => 0)
    const tickersTrace = {
      marker: {
        color: colors.gz,
        opacity: 0.8,
        symbol: 'line-ns-open',
      },
      mode: 'markers',
      type: 'scatter',
      x: ySamples,
      y: ySamples,
      yaxis: 'y2',
      showlegend: false,
    };

    const layout = {
      margin: {
        r: 1,
        t: 5,
        b: 5,
        l: 1
      },
      plot_bgcolor: 'rgba(0, 0, 0, 0)',
      paper_bgcolor: 'rgba(0, 0, 0, 0)',
      bargap: 0,
      xaxis: {
        range: [-3.5, 3.5]
      },
      yaxis: {
        range: [0, 0.32],
        domain: [0.15, 1]
      },
      yaxis2: {
        range: [-3.5, 3.5],
        domain: [0, 0.1],
        showgrid: false,
        zerolinecolor: '#ccc',
      },
      barmode: 'overlay',
      showlegend: true,
      legend: {
        x: 0,
        y: 0.9,
        xanchor: 'left',
      },
      ...defaultLayout,
      ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [targetTrace, outputTrace, tickersTrace], layout, {
      displayModeBar: false,
      staticPlot: true,
    });

    return async fakeOutputsSync => {
      const {
        x: outputX,
        y: outputY
      } = await histogram({
        min: -3.5,
        max: 3.5,
        numBins: 20,
        data: fakeOutputsSync,
        normalized: true,
      });

      Plotly.animate(containerEl, {
        data: [{
          x: outputX,
          y: outputY
        }, {
          x: targetX,
          y: targetY
        }, {
          x: fakeOutputsSync.slice(0, ySampleSize)
        }],
        traces: [1, 0, 2],
      }, {
        transition: {
          duration: animDuration,
          easing: 'ease-in'
        },
        frame: {
          duration: animDuration
        }
      });
    };
  }

  function initLossUI() {
    const [containerEl, layoutSize] = prepareDiagramElement({
      svgId: 'svg-object',
      textContent: 'loss-box',
    });

    const dLossTrace = {
      y: [],
      type: 'scatter',
      mode: 'lines',
      opacity: 0.8,
      line: {
        width: 4,
        color: colors.dLoss
      },
      name: '$D\ Loss$',
    };

    const gLossTrace = {
      y: [],
      type: 'scatter',
      mode: 'lines',
      opacity: 0.8,
      line: {
        width: 4,
        color: colors.gLoss
      },
      name: '$G\ Loss$',
    };

    const layout = {
      margin: {
        r: 0,
        t: 0,
        b: 0,
        l: 0
      },
      plot_bgcolor: 'rgba(0, 0, 0, 0)',
      paper_bgcolor: 'rgba(0, 0, 0, 0)',
      xaxis1: {
        ...hideGridAx
      },
      yaxis1: {
        ...hideGridAx
      },
      showlegend: true,
      legend: {
        x: 0,
        y: 1,
        xanchor: 'left',
        bgcolor: 'rgba(255,255,255,0.4)',
      },
      ...defaultLayout,
      ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [dLossTrace, gLossTrace], layout, {
      displayModeBar: false,
      staticPlot: true,
    });

    return (dLoss, gLoss) => {
      Plotly.extendTraces(containerEl, {
        y: [
          [dLoss],
          [gLoss]
        ]
      }, [0, 1], 50);
      // monkeyPatchSVGContext('svg-object', () => {});
    };
  }

  function initDBoxUI(x, y) {
    const [containerEl, layoutSize] = prepareDiagramElement({
      svgId: 'svg-object',
      textContent: 'd-box',
    });

    const trace = {
      x: x,
      y: y,
      type: 'scatter',
      mode: 'lines',
      line: {
        width: 5,
        color: colors.d
      }
    };

    const layout = {
      margin: {
        r: 1,
        t: 1,
        b: 20,
        l: 1
      },
      plot_bgcolor: 'rgba(0, 0, 0, 0)',
      paper_bgcolor: 'rgba(0, 0, 0, 0)',
      xaxis: {
        range: [-3, 3]
      },
      // yaxis: { range: [0, 1] },
      ...defaultLayout,
      ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [trace], layout, {
      displayModeBar: false,
      staticPlot: true,
      responsive: true,
    });

    return ySynced => {
      Plotly.restyle(containerEl, {
        y: ySynced
      }, 0);
    };
  }

  async function initDOutputUI(dx, dgz) {
    const [containerEl, layoutSize] = prepareDiagramElement({
      svgId: 'svg-object',
      textContent: 'dgz-box',
    });

    const {
      x: dxX,
      y: dxY,
    } = await histogram({
      data: dx,
      min: 0,
      max: 1,
      numBins: 20,
      normalized: true
    });

    const {
      x: dgzX,
      y: dgzY,
    } = await histogram({
      data: dgz,
      min: 0,
      max: 1,
      numBins: 20,
      normalized: true
    });

    const dxTrace = {
      x: dxX,
      y: dxY,
      marker: {
        color: colors.dx
      },
      type: 'bar',
      opacity: 0.8,
      name: '$D(X)$',
    };

    const dgzTrace = {
      x: dgzX,
      y: dgzY,
      marker: {
        color: colors.dgz
      },
      type: 'bar',
      opacity: 0.8,
      name: '$D(G(z))$',
    };

    const layout = {
      margin: {
        r: 1,
        t: 5,
        b: 5,
        l: 1
      },
      plot_bgcolor: 'rgba(0, 0, 0, 0)',
      paper_bgcolor: 'rgba(0, 0, 0, 0)',
      xaxis: {
        range: [0.01, 0.99]
      },
      yaxis: {
        range: [0, 0.5],
        domain: [0.08, 1]
      },
      barmode: 'overlay',
      showlegend: true,
      bargap: 0,
      legend: {
        x: 0,
        y: 1,
        xanchor: 'left',
      },
      ...defaultLayout,
      ...layoutSize,
    };

    Plotly.purge(containerEl);

    Plotly.newPlot(containerEl, [dxTrace, dgzTrace], layout, {
      displayModeBar: false,
      staticPlot: true,
    });

    return async (dx, dgz) => {
      const [{
        x: dxX,
        y: dxY,
      }, {
        x: dgzX,
        y: dgzY,
      }] = await Promise.all([
        histogram({
          data: dx,
          min: 0,
          max: 1,
          numBins: 20,
          normalized: true
        }),
        histogram({
          data: dgz,
          min: 0,
          max: 1,
          numBins: 20,
          normalized: true
        })
      ]);

      Plotly.animate(containerEl, {
        data: [{
          x: dxX,
          y: dxY
        }, {
          x: dgzX,
          y: dgzY
        }],
        traces: [0, 1],
      }, {
        transition: {
          duration: animDuration,
          easing: 'ease-in'
        },
        frame: {
          duration: animDuration
        }
      });
    };
  }

  return {
    initInputDataUI,
    initGANViewUI,
    initGANOutputUI,
    initLossUI,
    initDBoxUI,
    initDOutputUI,
  };
});
