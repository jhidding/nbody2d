# 2D PM n-body code for cosmology

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
