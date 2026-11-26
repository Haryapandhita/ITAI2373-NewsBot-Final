# Deployment Instructions (quick version)

Full details: [`deployment_guide.md`](deployment_guide.md).

**Google Colab (what the instructor should use):**
1. Open `notebooks/07_System_Integration.ipynb` on GitHub → *Open in Colab*.
2. *Optional:* Runtime → Change runtime type → **T4 GPU** (enables transformer models faster).
3. Runtime → **Run all**. The first cell clones the repo, installs `requirements.txt` + spaCy models and downloads the data. First build ≈ 8–10 min on CPU; cached afterwards.

**Local:**
```bash
git clone https://github.com/Haryapandhita/ITAI2373-NewsBot-Final.git
cd ITAI2373-NewsBot-Final/ITAI2373-NewsBot-Final
pip install -r requirements.txt
python -m spacy download en_core_web_sm && python -m spacy download xx_ent_wiki_sm
python -m pytest -q
python scripts/run_experiments.py      # rebuilds everything and all results/figures
```

No API keys are required. Optional switches are listed in `config/api_keys_template.txt`.
