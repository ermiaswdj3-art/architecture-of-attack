# The Architecture of Attack: code and evidence ledger

Code and the evidence ledger for the book *The Architecture of Attack: What Survived the Test* by Ermias Wedaje Tegbaru.

**This repository contains code only. It contains no data.** The scripts download the open data from its owners when you run `download_data.py`.

## Data and credits
- Event data: **StatsBomb Open Data**, 2018 FIFA World Cup (https://github.com/statsbomb/open-data). Data source: StatsBomb. Please read StatsBomb's user agreement before using the data.
- Tracking data: two anonymised sample matches published by **Metrica Sports** (https://github.com/metrica-sports/sample-data).
These organisations are not responsible for the analysis or conclusions here.

## Run it
```
pip install -r requirements.txt
python download_data.py          # fetches the data from its owners
python -m pytest tests.py        # 8 checks, including the goal totals
python check_goals.py
python ami_multi.py              # the index, all 64 matches
python validate_outcomes.py      # Chapter 10
python test_30s_robust.py        # Chapter 11
python place_vs_speed.py         # Chapter 12
python strata_auc.py             # Table 12.1
python hex_vs_square.py          # hexagons against squares
python pc_oos.py                 # Chapter 16
python paired_bootstrap.py       # pitch control against nearest-opponent
```
Results can differ in the last decimal place on different machines or library versions.

`Evidence_Ledger.md` lists every tested claim in the book and the script behind it.

## Licence
Code: MIT (see LICENSE). The data belongs to its owners and is not included.
