<p align="center"> 
  <img src="assets/screen_capture_h400_15fps_171118.gif">
</p>

### Simple analysis and visualization of patch clamp electrophysiology data in Python

This repo contains code for processing and visualization of current clamp electrophysiology data in Python
- Extract electrophysiology measurements from current clamp data (pClamp) using [stfio](https://github.com/neurodroid/stimfit) and [AllenSDK](http://alleninstitute.github.io/AllenSDK/)
- Animated plots of membrane voltage in response to current injection ([current_clamp.py](current_clamp.py))
- Automatic analysis pipeline using [Datajoint.io](https://github.com/datajoint/datajoint-python) to combine analysis results with multiple levels of metadata (animals, cells, analysis parameters, etc) ([schema_ephys.py](schema_ephys.py))
- Interactive data visualization using [Dash](https://medium.com/@plotlygraphs/introducing-dash-5ecf7191b503) and [Plot.ly](https://plot.ly/python/) ([iclamp_app.py](visualization/iclamp_app.py))
