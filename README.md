# Link-Traced Amplification: How Telegram Redistributes YouTube Political Content at Scale

This repository contains the code used to produce the results in our paper, accepted at **ACM Hypertext (HT '26)**, September 14–18, 2026, London, England.

> **Abstract.** Telegram is a lightly moderated platform hosting, among others, fringe and politically extreme communities, many of which migrated after being deplatformed from mainstream social media. While prior research has mainly analyzed political debate on Telegram through in-platform text, less attention has been paid to how external media content flows into and circulates within this ecosystem. We study the circulation of YouTube videos shared in 43,000 public Telegram chats surrounding the 2024 U.S. presidential election, analyzing 686,625 English-language videos as traces of cross-platform connectivity. Using topic modeling, supervised classification, and multi-dimensional toxicity measures, we characterize which narratives are amplified, how they diffuse by analyzing reach, recirculation, persistence, and transfer time, and whether toxicity is related to amplification. We find that Telegram functions as an agenda-redistribution layer for YouTube political content. Political videos diffuse in bursty, short-lived patterns that track high-salience events. Toxicity is only weakly associated with reach; instead, more toxic content tends to persist longer within narrower thematic circuits, reinforcing segmented information environments.

---

## Citation

If you use this code or refer to our findings, please cite:

*(Update with the final DOI once available in the ACM Digital Library.)*

---

## Repository Structure

```
US-Election-On-Telegram/
├── data/            # Intermediate and processed data (see "Data" section below)
├── figures/         # Generated plots used in the paper (Figures 1, 3–9)
├── notebooks/        # End-to-end pipeline, in numbered execution order (see table below)
├── PreProcessing/    # Text pre-processing utilities (lemmatization, stopword removal, etc.)
├── src/              # Standalone scripts for toxicity inference and classification
│   ├── classification_models/   # Random Forest / macro-topic classifier training and inference
│   ├── perspective_mat.py       # Helper functions for the Perspective API toxicity matrix
│   ├── run_detoxify.py          # Batch toxicity inference using Detoxify
│   └── run_perspective.py       # Batch toxicity inference using the Perspective API
├── lid.176.ftz        # fastText pretrained language-identification model (used for English filtering)
└── README.md
```

---

## Methodology → Code Mapping

The notebooks are numbered to follow the pipeline described in **Section 3** of the paper. Use this table as a guide to find the code behind each step.

| Paper section | Step | Notebook(s) |
|---|---|---|
| 3.1 – Dataset | Initial exploration of the Blas et al. Telegram dataset | `01_Telegram_dataset_analysis.ipynb` |
| 3.1 | Splitting/saving data by month | `02_Save_data_month.ipynb`, `03_Analysis_month.ipynb` |
| 3.1 | Basic descriptive analysis | `04_Basic_Analysis.ipynb` |
| 3.1.1 – URL-derived dataset construction | Extracting all URLs from messages | `05_Links.ipynb` |
| 3.1.2 / 3.1.3 – Domain filtering & YouTube video ID selection | Filtering YouTube video URLs | `06_Filtering_Yt_Videos.ipynb` |
| 3.1.4 – Enrichment via YouTube Data API | Retrieving title, description, statistics | `07_Extracting_infos_yt.ipynb` |
| 3.1.5 – Final language filtering | Language detection (English-only), using `lid.176.ftz` | `08_English_videos.ipynb` |
| 3.1.6 – Data pre-processing | Lemmatization, stopword removal, normalization | `09_Pre_processing.ipynb`, `PreProcessing/` |
| 3.2 – Topic Modeling | Hyperparameter search, sampling, evaluation metrics | `10_Metrics_analysis.ipynb`, `11_TM_parameters.ipynb`, `12_TM_sample.ipynb`, `13_Best_TM.ipynb` |
| 3.2 | Final BERTopic configuration on 10% sample | `14_TM_parameters.ipynb`, `15_TM_sample.ipynb`, `16_Best_TM.ipynb` |
| 3.2 | Merging topic-modeling outputs | `14_Merging_tables.ipynb`, `18_Merging_tables.ipynb` |
| 3.2 | Auxiliary regression/interest analysis | `17_Regression_interest.ipynb` |
| 3.2 | Temporal exploration of topics | `19_Time_analysis.ipynb` |
| 3.3 – Classifying the remaining videos | Supervised classifier (kNN baseline, Random Forest) | `20_KNN.ipynb`, `21_Evaluating_classification_models.ipynb`, `src/classification_models/` |
| 3.2 / 3.3.1 – Macro-topic construction & summarization | Grouping BERTopic topics into macro-topics | `22_Macrotopics.ipynb` |
| 3.3 | Analyzing macro-topic distribution | `23_Macrotopics_analysis.ipynb`, `24_Topics_analysis.ipynb` |
| 3.4 – Toxicity estimation | Batch scoring with Perspective API / Detoxify | `src/run_perspective.py`, `src/run_detoxify.py`, `src/perspective_mat.py` |
| 3.4 / 4.1.2 | Relating toxicity to macro-topics | `23_Macrotopics_perspective.ipynb` |
| 4 – Results | Final metrics, figures, and tables assembly | `26.ipynb`, `27_Organizando_Resultados.ipynb`, `28.ipynb` |

> **Note:** Some notebooks share the same prefix number or exist as "copy" versions (e.g., `22_Macrotopics copy.ipynb`). These reflect iterations made during development. The versions referenced above (and listed without "copy" in the filename) correspond to the final results reported in the paper. We plan to clean these up in a future revision of the repository — see [Reproducibility Notes](#reproducibility-notes).

---

## Data

Due to size and platform Terms of Service, we do not redistribute the raw Telegram message dump or full YouTube metadata in this repository.

- **Raw Telegram data**: originally released by Blas et al., *"Unearthing a Billion Telegram Posts about the 2024 U.S. Presidential Election"* — see their [dataset repository](https://github.com/leonardo-blas/usc-tg-24-us-election).
- **YouTube metadata**: retrieved via the [YouTube Data API v3](https://developers.google.com/youtube/v3), subject to YouTube's Terms of Service. Users wishing to reproduce this step will need their own API key.
- **Toxicity scores**: computed via the [Perspective API](https://perspectiveapi.com/) (primary) and [Detoxify](https://github.com/unitaryai/detoxify) (for comparison), both requiring separate setup (see below).

The `data/` folder in this repository contains intermediate, non-identifying, aggregate outputs (e.g., macro-topic assignments, toxicity score tables) needed to reproduce the figures and tables in the paper.

---

## Setup

```bash
git clone https://github.com/MateusoBrito/US-Election-On-Telegram.git
cd US-Election-On-Telegram
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducibility Notes

This codebase reflects the iterative, exploratory research process behind the paper rather than a polished, one-command pipeline. Notebooks are numbered to approximate the chronological/logical order of the methodology in Section 3, but some steps (particularly topic modeling hyperparameter search) involved multiple rounds of experimentation, reflected in duplicate-numbered or "copy" notebooks.

We are working on a cleaned-up version of this pipeline. Planned improvements:
- [ ] Add `requirements.txt` / `environment.yml`
- [ ] Remove duplicate/"copy" notebooks or clearly mark superseded versions
- [ ] Rename unclear notebooks (`26.ipynb`, `28.ipynb`)
- [ ] Add `.gitignore` for `__pycache__/` and other build artifacts
- [ ] Consider Git LFS for `lid.176.ftz` if repository size becomes an issue

If you run into issues reproducing a specific result, please open an issue — we're happy to help clarify any step.

---

## License

This source code is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
---

## Authors

- Mateus Brito — Universidade Federal de São João del Rei
- Ester Souza — Universidade Federal de São João del Rei
- Thiago Braga — Universidade Federal de São João del Rei
- Giordano Paoletti — Politecnico di Torino
- Luca Vassio — Politecnico di Torino
- Jussara M. Almeida — Universidade Federal de Minas Gerais
- Leonardo Rocha — Universidade Federal de São João del Rei

For questions, contact: mateusdeoliveirabritoo@aluno.ufsj.edu.br