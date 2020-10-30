# 2D PM n-body code for cosmology
[![Entangled badge](https://img.shields.io/badge/entangled-Use%20the%20source!-%2300aeff)](https://entangled.github.io/)

![A sample simulation, three time steps; the bottom row is a zoom-in.](figures/x.collage.png)

## Install
This code has the following dependencies:

- Python3
- Gnuplot
- `pip install -r requirements.txt` will install `numpy` and `scipy`.

## Run
From project directory run

        python -m nbody.nbody

Progress on simulation is plotted with Gnuplot. Results end up in `./data` directory.

## Development
I'm in the process of making this code literate. To edit the code, have [Entangled](https://entangled.github.io/) installed and running:

        entangled daemon

Documentation is currently minimal. The literate code can be viewed in rendered form [on the github.io pages](https://jhidding.github.io/nbody2d).
