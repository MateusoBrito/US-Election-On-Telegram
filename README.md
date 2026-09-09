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
├── docs/            # Project documentation and supplementary files
├── figures/         # Generated plots used in the paper
├── notebooks/       # Clean, end-to-end pipeline in execution order (01 to 14)
│   └── archive/     # Legacy & exploratory notebooks from development iterations
├── reports/         # Generated analysis reports
├── src/             # Standalone Python scripts & modules
│   ├── PreProcessing/         # Text pre-processing utilities (lemmatization, stopword removal, etc.)
│   ├── classification_models/ # Macro-topic classifier training and inference
│   ├── meu_bertopic.py        # Helper routines for BERTopic customization
│   ├── perspective_mat.py     # Perspective API toxicity helpers
│   ├── run_detoxify.py        # Batch toxicity inference using Detoxify
│   └── run_perspective.py     # Batch toxicity inference using Perspective API
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Methodology → Code Mapping

The notebooks have been refactored into a clean 14-step pipeline that follows Section 3 of the paper.

| Paper section | Step | Notebook(s) / Code |
|---|---|---|
| 3.1 – Dataset | Initial dataset architecture & structure | `01_Dataset Architecture.ipynb` |
| 3.1 | Basic descriptive dataset exploration | `02_Basic_Dataset_Exploration1.ipynb` |
| 3.1.1 – URL-derived dataset construction | Social media link distribution & study | `03_Social_Media_Links_Study.ipynb` |
| 3.1.2 / 3.1.3 – Domain filtering & YouTube selection | Filtering YouTube video URLs | `04_Filtering_Yt_Videos.ipynb` |
| 3.1.4 – Enrichment via YouTube Data API | Retrieving titles, descriptions, statistics | `05_Extracting_infos_yt.ipynb` |
| 3.1.5 – Final language filtering | Language detection (English-only) | `06_English_videos.ipynb` |
| 3.1.6 – Data pre-processing | Lemmatization, stopword removal, normalization | `07_Pre_processing.ipynb`, `src/PreProcessing/` |
| 3.2 – Topic Modeling | BERTopic hyperparameter search & evaluation | `08_TM_parameters.ipynb` |
| 3.2 | Topic modeling sampling (10% sample) | `09_TM_sample.ipynb` |
| 3.2 | Optimal BERTopic model training & assignment | `10_Best_TM.ipynb` |
| 3.2 | Merging topic outputs with full dataset | `11_Merging_tables.ipynb` |
| 3.3 – Classifying remaining videos | Supervised classifier for macro-topics | `12_Macrotopic_Classifier.ipynb`, `src/classification_models/` |
| 3.4 & 4 – Results data preparation | Consolidating topics, toxicity, and statistics | `13_Preparing_Results_Data.ipynb` |
| 4 – Results | Final metrics, figures, and statistical plots assembly | `14_Results_Figures_and_Stats.ipynb` |

> **Note:** The notebook pipeline was streamlined from original exploratory versions into a clean sequential structure (`01` through `14`). Legacy, exploratory, and duplicate notebooks from early iterations are stored in `notebooks/archive/`.

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

The codebase pipeline is organized sequentially from step `01` to `14` to replicate the paper's methodology. 

Completed refactoring & repository improvements:
- [x] Streamline and standardize notebook pipeline (01–14 sequential order)
- [x] Move legacy/exploratory notebooks to `notebooks/archive/`
- [x] Organize text pre-processing module under `src/PreProcessing/`
- [x] Include `requirements.txt`
- [ ] Add `.gitignore` for `__pycache__/` and build artifacts

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