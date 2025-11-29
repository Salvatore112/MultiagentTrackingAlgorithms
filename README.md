# System for analysis of multi-agent algorithms for objects tracking
[![CI: Linter](https://github.com/Salvatore112/MultiagentTrackingAlgorithms/actions/workflows/lint.yml/badge.svg)](https://github.com/Salvatore112/BaseConfigGen/actions/workflows/ci.yml)
[![CI: Tests](https://github.com/Salvatore112/MultiagentTrackingAlgorithms/actions/workflows/test.yml/badge.svg)](https://github.com/Salvatore112/BaseConfigGen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This system is designed to analyze and compare different multi-agent tracking algorithms

## Requirements

- **Python 3.12+**

## Build
```bash
git clone https://github.com/Salvatore112/MultiagentTrackingAlgorithms.git
cd MultiagentTrackingAlgorithms
python -m venv .venv
. .venv/bin/activate
pip install -r req.txt
```

## Launch
```bash
python manage.py migrate
python manage.py runserver
```
The application will start at `http://127.0.0.1:8000/` by default.

## Usage
After you start the server, you can go to `http://127.0.0.1:8000/`. The setup page will open where you can choose algorithms and configure the simulation.
In this example there would be 3 sensors observing 4 targets. The simulation will generate 200 seconds of the target's movements and their distances to each sensor at each moment. 
The distances will be noise. The noise is uniformly genereated from the interval [-0.5, 0.5]
<img width="1327" height="778" alt="image" src="https://github.com/user-attachments/assets/1aa033da-e97e-4e6e-89b2-dafdeb913ba2" />
After everything is configured, the user can start the simulation. The chosen algorithms will be ran and the plots of the estimates of each sensor will be shown as well as the plot of the error evolution. You can then see how each sensor observed targets and how its error changed.
<img width="1256" height="842" alt="image" src="https://github.com/user-attachments/assets/d2581dc3-cdc2-42bf-a9fc-68f206279a67" />

## Contact us

If you want to help the project you may open a pull request. You can contact [@me](https://t.me/unacerveza1) to ask any questions or give an advice
