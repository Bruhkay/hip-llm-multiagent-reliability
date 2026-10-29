# Results

Raw outputs produced by the notebooks:

| File | Produced by | Content |
|---|---|---|
| `safety_eval_raw.csv` | `02_safety_agent_tmr_homogeneous.ipynb` | Per-run, per-replica outputs (Gemini ×3) |
| `safety_eval_raw_3models.csv` | `03_safety_agent_tmr_heterogeneous.ipynb` | Per-run, per-replica outputs (GPT / Gemini / Qwen) |
| `tmr_eval_homogeneous.csv` | notebook 02 (`tmr_eval.csv`) | Majority-vote (TMR) vs. single-replica scoring |
| `tmr_eval_heterogeneous.csv` | notebook 03 (`tmr_eval.csv`) | Majority-vote (TMR) vs. single-replica scoring |

Notebooks 02 and 03 both write `tmr_eval.csv`; the files are renamed here to keep them apart.
