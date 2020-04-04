---
layout: post
title: "Notebook template post"
date: 2020-01-01 21:39:01 +0200
categories: neural networks, cppn, art
---
<p class='red'>red</p>

# Test

ala bala

- test
- test4

$\sum_{i=1}^{k+1}i$


```javascript
%%javascript

element.text('test')
```


    <IPython.core.display.Javascript object>



```javascript
%%javascript

require.undef('aaa');
define('aaa', [], function() {
    return {
        a: () => alert(5),
    };
});
```


    <IPython.core.display.Javascript object>



```javascript
%%javascript

require(['aaa'], function(a) {
    a.a();
});
```


    <IPython.core.display.Javascript object>



```python
import numpy as np
import matplotlib.pyplot as plt, mpld3

mpld3.display()
mpld3.enable_notebook()
```


```python
plt.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
plt.imshow(np.random.rand(10, 10, 3));
```




<style>

</style>

<div id="fig_el2450139619407274576477872604"></div>
<script>
function mpld3_load_lib(url, callback){
  var s = document.createElement('script');
  s.src = url;
  s.async = true;
  s.onreadystatechange = s.onload = callback;
  s.onerror = function(){console.warn("failed to load library " + url);};
  document.getElementsByTagName("head")[0].appendChild(s);
}

if(typeof(mpld3) !== "undefined" && mpld3._mpld3IsLoaded){
   // already loaded: just create the figure
   !function(mpld3){

       mpld3.draw_figure("fig_el2450139619407274576477872604", {"width": 432.0, "height": 288.0, "axes": [{"bbox": [0.26083333333333336, 0.125, 0.5033333333333333, 0.755], "xlim": [-0.5, 9.5], "ylim": [9.5, -0.5], "xdomain": [-0.5, 9.5], "ydomain": [9.5, -0.5], "xscale": "linear", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 7, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}, {"position": "left", "nticks": 7, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el2450139619407276176", "lines": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el2450139619403366672", "color": "#000000", "linewidth": 1.5, "dasharray": "none", "alpha": 1, "zorder": 2, "drawstyle": "default"}], "paths": [], "markers": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el2450139619403366672pts", "facecolor": "#000000", "edgecolor": "#FFFFFF", "edgewidth": 5, "alpha": 1, "zorder": 2, "markerpath": [[[-10.0, 10.0], [10.0, 10.0], [10.0, -10.0], [-10.0, -10.0]], ["M", "L", "L", "L", "Z"]]}], "texts": [], "collections": [], "images": [{"data": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAABHNCSVQICAgIfAhkiAAAAaNJREFUGJUFwVtIUwEcwOHfOduhzEtlCZKnQHdE6SHMwBtdEC2SKAwtC5IYRNTDYC8OZBCCQeANiswUXAYlPlhCk2pIBEUSmjqbIXiWWBZsOCosanPz/Ps+pfCxLuMVYe5PbxJfKEErijJ0O8DD7yZOrY720TQyLT+KfuelVJ4tJ5x8QUgLU5LawVzS4mbjDNW9zcSWSrnUkQt9+CW9oVJG418l68CAvHt6SP6NrMqY/5x86rWLd3Ja8i8GRH2ttxBMplE462bne4Xgug9TPc015yMu7Osn65aXq1s8sGz0S92DnzI8OyULv8fllXVM9lpv5YjRJbkFK2IGOsX5+ZfY1oy/beWLczSph+neM0PEc53jxnYi0SjZOSZa4h6ZykkUh+uNVNj6iOXpHLSlyAstUxCr4q6vjaWV3dR63dRaY6iJsgk2OnbREPQwuPEch3oeT8iH+cRNmTTyTB1h/xcX9uKebI465rFah8kpMshwNRH6sY3SExHiN6pY37yCvvoBdSq+yLdoBomt7Wh/PlJzJoU9PZ/6yXkCrTrJoRYur53iP6z2se/kdsQXAAAAAElFTkSuQmCC", "extent": [-0.5, 9.5, 9.5, -0.5], "coordinates": "data", "alpha": null, "zorder": 0, "id": "el2450139619407277968"}], "sharex": [], "sharey": []}], "data": {"data01": [[0.0, 3.0], [1.0, 1.0], [2.0, 4.0], [3.0, 1.0], [4.0, 5.0]]}, "id": "el2450139619407274576", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}]});
   }(mpld3);
}else if(typeof define === "function" && define.amd){
   // require.js is available: use it to load d3/mpld3
   require.config({paths: {d3: "https://mpld3.github.io/js/d3.v3.min"}});
   require(["d3"], function(d3){
      window.d3 = d3;
      mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.3.js", function(){

         mpld3.draw_figure("fig_el2450139619407274576477872604", {"width": 432.0, "height": 288.0, "axes": [{"bbox": [0.26083333333333336, 0.125, 0.5033333333333333, 0.755], "xlim": [-0.5, 9.5], "ylim": [9.5, -0.5], "xdomain": [-0.5, 9.5], "ydomain": [9.5, -0.5], "xscale": "linear", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 7, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}, {"position": "left", "nticks": 7, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el2450139619407276176", "lines": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el2450139619403366672", "color": "#000000", "linewidth": 1.5, "dasharray": "none", "alpha": 1, "zorder": 2, "drawstyle": "default"}], "paths": [], "markers": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el2450139619403366672pts", "facecolor": "#000000", "edgecolor": "#FFFFFF", "edgewidth": 5, "alpha": 1, "zorder": 2, "markerpath": [[[-10.0, 10.0], [10.0, 10.0], [10.0, -10.0], [-10.0, -10.0]], ["M", "L", "L", "L", "Z"]]}], "texts": [], "collections": [], "images": [{"data": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAABHNCSVQICAgIfAhkiAAAAaNJREFUGJUFwVtIUwEcwOHfOduhzEtlCZKnQHdE6SHMwBtdEC2SKAwtC5IYRNTDYC8OZBCCQeANiswUXAYlPlhCk2pIBEUSmjqbIXiWWBZsOCosanPz/Ps+pfCxLuMVYe5PbxJfKEErijJ0O8DD7yZOrY720TQyLT+KfuelVJ4tJ5x8QUgLU5LawVzS4mbjDNW9zcSWSrnUkQt9+CW9oVJG418l68CAvHt6SP6NrMqY/5x86rWLd3Ja8i8GRH2ttxBMplE462bne4Xgug9TPc015yMu7Osn65aXq1s8sGz0S92DnzI8OyULv8fllXVM9lpv5YjRJbkFK2IGOsX5+ZfY1oy/beWLczSph+neM0PEc53jxnYi0SjZOSZa4h6ZykkUh+uNVNj6iOXpHLSlyAstUxCr4q6vjaWV3dR63dRaY6iJsgk2OnbREPQwuPEch3oeT8iH+cRNmTTyTB1h/xcX9uKebI465rFah8kpMshwNRH6sY3SExHiN6pY37yCvvoBdSq+yLdoBomt7Wh/PlJzJoU9PZ/6yXkCrTrJoRYur53iP6z2se/kdsQXAAAAAElFTkSuQmCC", "extent": [-0.5, 9.5, 9.5, -0.5], "coordinates": "data", "alpha": null, "zorder": 0, "id": "el2450139619407277968"}], "sharex": [], "sharey": []}], "data": {"data01": [[0.0, 3.0], [1.0, 1.0], [2.0, 4.0], [3.0, 1.0], [4.0, 5.0]]}, "id": "el2450139619407274576", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}]});
      });
    });
}else{
    // require.js not available: dynamically load d3 & mpld3
    mpld3_load_lib("https://mpld3.github.io/js/d3.v3.min.js", function(){
         mpld3_load_lib("https://mpld3.github.io/js/mpld3.v0.3.js", function(){

                 mpld3.draw_figure("fig_el2450139619407274576477872604", {"width": 432.0, "height": 288.0, "axes": [{"bbox": [0.26083333333333336, 0.125, 0.5033333333333333, 0.755], "xlim": [-0.5, 9.5], "ylim": [9.5, -0.5], "xdomain": [-0.5, 9.5], "ydomain": [9.5, -0.5], "xscale": "linear", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 7, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}, {"position": "left", "nticks": 7, "tickvalues": null, "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": false}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el2450139619407276176", "lines": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el2450139619403366672", "color": "#000000", "linewidth": 1.5, "dasharray": "none", "alpha": 1, "zorder": 2, "drawstyle": "default"}], "paths": [], "markers": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el2450139619403366672pts", "facecolor": "#000000", "edgecolor": "#FFFFFF", "edgewidth": 5, "alpha": 1, "zorder": 2, "markerpath": [[[-10.0, 10.0], [10.0, 10.0], [10.0, -10.0], [-10.0, -10.0]], ["M", "L", "L", "L", "Z"]]}], "texts": [], "collections": [], "images": [{"data": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAABHNCSVQICAgIfAhkiAAAAaNJREFUGJUFwVtIUwEcwOHfOduhzEtlCZKnQHdE6SHMwBtdEC2SKAwtC5IYRNTDYC8OZBCCQeANiswUXAYlPlhCk2pIBEUSmjqbIXiWWBZsOCosanPz/Ps+pfCxLuMVYe5PbxJfKEErijJ0O8DD7yZOrY720TQyLT+KfuelVJ4tJ5x8QUgLU5LawVzS4mbjDNW9zcSWSrnUkQt9+CW9oVJG418l68CAvHt6SP6NrMqY/5x86rWLd3Ja8i8GRH2ttxBMplE462bne4Xgug9TPc015yMu7Osn65aXq1s8sGz0S92DnzI8OyULv8fllXVM9lpv5YjRJbkFK2IGOsX5+ZfY1oy/beWLczSph+neM0PEc53jxnYi0SjZOSZa4h6ZykkUh+uNVNj6iOXpHLSlyAstUxCr4q6vjaWV3dR63dRaY6iJsgk2OnbREPQwuPEch3oeT8iH+cRNmTTyTB1h/xcX9uKebI465rFah8kpMshwNRH6sY3SExHiN6pY37yCvvoBdSq+yLdoBomt7Wh/PlJzJoU9PZ/6yXkCrTrJoRYur53iP6z2se/kdsQXAAAAAElFTkSuQmCC", "extent": [-0.5, 9.5, 9.5, -0.5], "coordinates": "data", "alpha": null, "zorder": 0, "id": "el2450139619407277968"}], "sharex": [], "sharey": []}], "data": {"data01": [[0.0, 3.0], [1.0, 1.0], [2.0, 4.0], [3.0, 1.0], [4.0, 5.0]]}, "id": "el2450139619407274576", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}]});
            })
         });
}
</script>


```python
import matplotlib.pyplot as plt, mpld3
mpld3.enable_notebook()

plt.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)
plt.imshow(np.random.rand(10, 10, 3))
mpld3.display()
```


```python
import pandas as pd

pd.DataFrame({'a':[1,2], 'b':[3,4]})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
{'a':2}.get('b', False)
```




    False



---


```python

```
