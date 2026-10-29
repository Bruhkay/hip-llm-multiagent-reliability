# Data

## `ground_truth.csv`
Expected harmonised GHS hazard statements (H-codes) for the nine evaluation substances, taken from the ECHA Substance Infocard pages (`https://echa.europa.eu/substance-information/-/substanceinfo/<id>`) at the time of the experiments. Used to score the safety agent's outputs.

## ECHA page snapshots
The safety-agent experiments (notebooks 02 and 03) run on saved HTML snapshots of the ECHA pages so that every run sees identical input.
The snapshots are in [`echa_pages/`](echa_pages/) (on Kaggle they were attached as the dataset `burkaytuntrk/echa-page`, mounted under `/kaggle/input`).

To run locally, point the notebooks at them:

```bash
export ECHA_HTML_DIR=data/echa_pages   # relative to the repo root
```

File names must keep the original format (`<Substance> 100.xxx.xxx.html`), because the substance label is parsed from the name.
