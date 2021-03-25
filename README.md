# 2D PM n-body code for cosmology
[![Entangled badge](https://img.shields.io/badge/entangled-Use%20the%20source!-%2300aeff)](https://entangled.github.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4158731.svg)](https://doi.org/10.5281/zenodo.4158731)

![A sample simulation, three time steps; the bottom row is a zoom-in.](docs/figures/x.collage.png)

## Install
This code has the following dependencies:

- Python3
- Gnuplot
- `pip install -r requirements.txt` will install `numpy` and `scipy`.

## Run
From project directory run

        python -m nbody.nbody

Progress on simulation is plotted with Gnuplot. Results end up in `./data` directory.

## Citation
If you use this code in a scientific publication, please cite it using this DOI:[10.5281/zenodo.4158731](https://doi.org/10.5281/zenodo.4158731).

## Development
I'm in the process of making this code literate. To edit the code, have [Entangled](https://entangled.github.io/) installed and running:

        entangled daemon

Documentation is currently minimal. The literate code can be viewed in rendered form [on the github.io pages](https://jhidding.github.io/nbody2d).
