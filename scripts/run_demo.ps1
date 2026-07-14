$ErrorActionPreference = "Stop"
python -m pip install -e .
python -m pytest
python -m prepayment demo --loans 3000 --seed 42 --output reports/demo
