# HIP-LLM Reliability Assessment of a Multi-Agent SMILES → ECHA Hazard Pipeline

Code accompanying the paper on reliability assessment of LLM-based multi-agent systems using **HIP-LLM** (Hierarchical Imprecise Probability) and Bayesian networks, with a case study on chemical hazard extraction.

The pipeline takes a molecule as a SMILES string and returns its harmonised GHS hazard statements (H-codes) from the European Chemicals Agency (ECHA). We measure the reliability of each agent, apply **Triple Modular Redundancy (TMR)** with majority voting to the safety agent, and propagate HIP-LLM reliability bounds through a Bayesian network (pyAgrum) to compare system-level failure probabilities with and without TMR.

## Pipeline

```
SMILES ──► Research agent ──► PixelRAG capture agent ──► Safety (vision) agent ──► H-codes
           (PubChem + web       (CDP browser, renders      (multimodal LLM reads
            search → ECHA page)  page into image tiles)     tiles, returns H-codes)
```

The agents are orchestrated with LangGraph. Shared code lives in [`src/agentic_smiles_hazard_utils.py`](src/agentic_smiles_hazard_utils.py).

## Repository layout

```
src/        agentic_smiles_hazard_utils.py – identity lookup, ECHA search, browser/PixelRAG capture, LLM helpers
notebooks/  experiments (see below)
data/       ground_truth.csv + notes on the ECHA page snapshots
results/    raw CSV outputs of the experiments
figures/    Bayesian-network graphs and plots
```

## Experiments

| Notebook | Experiment | Models | Output |
|---|---|---|---|
| `01_research_agent_eval.ipynb` | Repeated runs of the research agent on one SMILES (toluene, `Cc1ccccc1`) | gpt-5-mini | printed success/failure per run |
| `02_safety_agent_tmr_homogeneous.ipynb` | Safety agent, homogeneous TMR (3 identical replicas), 5 runs × 9 substances | gemini-2.5-flash ×3 | `safety_eval_raw.csv`, `tmr_eval.csv`, BN bounds |
| `03_safety_agent_tmr_heterogeneous.ipynb` | Safety agent, heterogeneous TMR (3 different models), 10 runs × 9 substances | gpt-5-mini, gemini-2.5-flash, qwen2.5vl:3b | `safety_eval_raw_3models.csv`, `tmr_eval.csv`, BN bounds, `bayesian_graphs.zip` |

The operational profile used for HIP-LLM is `{"single": 0.2, "multi": 0.8}`. The TMR Bayesian network is: pairwise AND nodes → 2-out-of-3 OR → system node with a voter failure probability of 1e-3; optimistic and pessimistic CPTs are built from the HIP-LLM lower/upper bounds.

## Setup

Python ≥ 3.10.

```bash
git clone https://github.com/<user>/hip-llm-multiagent-reliability.git
cd hip-llm-multiagent-reliability
pip install -r requirements.txt

# PixelRAG (pinned commit used in the paper)
git clone https://github.com/StarTrail-org/PixelRAG.git
cd PixelRAG && git checkout 7a3c3041bd37b1868d8573039ebdc60607ddb936 && cd ..
```

A Chromium/Chrome binary is needed for the capture agent (it is driven over the Chrome DevTools Protocol).

### API keys and endpoints

Set these as environment variables (on Kaggle, add them as *Secrets*; the code reads either):

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | research agent, GPT replica |
| `GEMINI_API_KEY` | Gemini replicas |
| `QWEN_BASE_URL` | OpenAI-compatible endpoint serving `qwen2.5vl:3b` (e.g. Ollama: `http://127.0.0.1:11434/v1`) |
| `QWEN_API_KEY` | optional, if the Qwen endpoint requires one |
| `ECHA_HTML_DIR` | folder with the ECHA page snapshots (default `/kaggle/input`; locally use `data/echa_pages`) |

Optional overrides: `OPENAI_MODEL`, `OPENAI_AGENT_MODEL`, `GEMINI_MODEL`, `QWEN_MODEL`.

### Running

The experiments were run on **Kaggle notebooks** and the notebooks keep Kaggle's default paths (`/kaggle/working` for outputs, `/kaggle/input` for inputs). To reproduce on Kaggle: upload `src/agentic_smiles_hazard_utils.py` and the ECHA snapshot dataset as inputs, add the secrets, and run the notebook top to bottom. Locally, run Jupyter from the repo root or from `notebooks/`; the loader cell finds `src/` automatically. Create `/kaggle/working` or change the output paths in the config cells.

Notebooks 02 and 03 resume from an existing raw CSV; set `FORCE_RERUN = True` to start fresh.

## Known limitations

- LLM outputs are non-deterministic; reruns give different numbers within the reported bounds.
- ECHA pages change over time. The safety-agent experiments therefore use saved snapshots (see `data/README.md`); the research agent queries the live web.
- In notebook 03 the Qwen replica receives only the first rendered tile, and the intended 800 px downscaling is skipped (`io.BytesIO` is referenced while only `BytesIO` is imported; the exception is caught silently). Results in the paper were produced with this behaviour, so it is left unchanged.

## Citation

```bibtex
@article{tbd2026hipllm,
  title   = {TBD},
  author  = {Tun{\c{c}}t{\"u}rk, Burkay and others},
  journal = {IEEE Software},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT – see [LICENSE](LICENSE). PixelRAG and HIP-LLM are separate projects under their own licenses.
