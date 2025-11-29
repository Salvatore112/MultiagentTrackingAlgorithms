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
