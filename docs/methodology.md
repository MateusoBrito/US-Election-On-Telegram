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