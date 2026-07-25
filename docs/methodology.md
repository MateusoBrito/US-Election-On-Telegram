## Methodology → Code Mapping

The notebooks are numbered to follow the pipeline described in **Section 3** of the paper. Use this table as a guide to find the code behind each step.

Step | Notebook(s) |
---|---|
Initial exploration of the Blas et al. Telegram dataset | `01_Dataset Architecture.ipnyb` |
Basic descriptive analysis | `02_Basic_Dataset_Exploration.ipynb`, `03_Social_Media_Links_Study.ipynb`  |
Filtering YouTube video URLs | `04_Filtering_Yt_Videos.ipynb` |
Retrieving title, description, statistics | `05_Extracting_infos_yt.ipynb`
Language detection (English-only) | `06_English_videos.ipynb` |
Pre processing| `07_Pre_processing.ipynb`, `PreProcessing/` |
Hyperparameter search, sampling, evaluation metrics | `10_Metrics_analysis.ipynb`, `11_TM_parameters.ipynb`, `12_TM_sample.ipynb`, `13_Best_TM.ipynb` |
Final BERTopic configuration on 10% sample | `14_TM_parameters.ipynb`, `15_TM_sample.ipynb`, `16_Best_TM.ipynb` |
Merging topic-modeling outputs | `14_Merging_tables.ipynb`, `18_Merging_tables.ipynb` |
Auxiliary regression/interest analysis | `17_Regression_interest.ipynb` |
Temporal exploration of topics | `19_Time_analysis.ipynb` |
Supervised classifier (kNN baseline, Random Forest) | `20_KNN.ipynb`, `21_Evaluating_classification_models.ipynb`, `src/classification_models/` |
Grouping BERTopic topics into macro-topics | `22_Macrotopics.ipynb` |
Analyzing macro-topic distribution | `23_Macrotopics_analysis.ipynb`, `24_Topics_analysis.ipynb` |
Batch scoring with Perspective API / Detoxify | `src/run_perspective.py`, `src/run_detoxify.py`, `src/perspective_mat.py` |
Relating toxicity to macro-topics | `23_Macrotopics_perspective.ipynb` |
Final metrics, figures, and tables assembly | `26.ipynb`, `27_Organizando_Resultados.ipynb`, `28.ipynb` |

> **Note:** Some notebooks share the same prefix number or exist as "copy" versions (e.g., `22_Macrotopics copy.ipynb`). These reflect iterations made during development. The versions referenced above (and listed without "copy" in the filename) correspond to the final results reported in the paper. We plan to clean these up in a future revision of the repository — see [Reproducibility Notes](#reproducibility-notes).

---