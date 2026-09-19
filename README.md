# From Port to Playlist: Acoustic Overlap and Timbral Divergence in Caribbean and West African Music

This repository accompanies the thesis *From Port to Playlist: Acoustic Overlap and Timbral Divergence in Caribbean and West African Music*, submitted by Mhea Bardouille in satisfaction of the requirements for the degree of Master of Science in Big Data and Business Analytics (STRUCTURALIA, 2026). The study constructs a corpus of 364 thirty-second commercial audio previews across four Caribbean genre traditions (Calypso/Soca, Reggae/Dancehall, Compas, Zouk) and four West African genre traditions (Afrobeats, Highlife, Kora, Juju), extracts 43 acoustic descriptors across tonal, temporal, dynamic, and timbral categories, retains 42 after near-zero-variance screening, and examines which Caribbean genres are most acoustically similar to the West African reference corpus, which features contribute most strongly to similarity and divergence, and how effectively supervised models separate the two regional corpora.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── corpus_metadata.csv
│   ├── raw/
│   │   └── audio/
│   └── processed/
│       ├── features.csv
│       ├── features_eda_clean.csv
│       ├── normality_test_results.csv
│       ├── feature_effect_sizes.csv
│       ├── dropped_features.txt
│       ├── table_5_1_test_selection_summary.csv
│       ├── table_5_2_concise_results.csv
│       ├── table_5_2_corpus_comparisons.csv
│       ├── table_5_3_genre_comparisons.csv
│       ├── table_5_4_posthoc_caribbean_vs_wa.csv
│       ├── acoustic_proximity_ranking.csv
│       ├── genre_similarity_profile.csv
│       ├── genre_similarity_matrix_full.csv
│       ├── feature_level_similarity.csv
│       ├── cluster_assignments.csv
│       ├── cluster_evaluation_summary.json
│       ├── table7b_class_metrics_comparison.csv
│       ├── table7b_overall_metrics_comparison.csv
│       ├── figure_5_1_volcano.png
│       ├── figure_5_2_significant_effect_sizes.png
│       ├── figure_5_2_significant_effect_sizes_updated.png
│       ├── figure_5_3_proximity_ranking.png
│       ├── figure_5_4_posthoc_heatmap.png
│       ├── figure_9_dendrogram_genre_profiles.png
│       └── figure10b_confusion_matrix_comparison.png
├── notebooks/
│   ├── 02_feature_extraction.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_statistical_comparison.ipynb
│   ├── 05_similarity_analysis.ipynb
│   └── 06_clustering_and_classification.ipynb
└── scripts/
    └── collect.py
```

## Environment setup

The dependency set has been verified with Python 3.13. scikit-learn 1.6.1 does not provide wheels for Python 3.14, so a Python 3.13 interpreter is required for the pinned environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Corpus retrieval requires Spotify Web API credentials. Copy the environment template and supply your own values:

```bash
cp .env.example .env
```

`SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` are the only required values. The iTunes Search API and the MusicBrainz API require no credentials.

The balanced class-weight sensitivity analysis reported in this study depends on the exact version of scikit-learn used at execution time. scikit-learn 1.6.1 is pinned in requirements.txt specifically to reproduce the reported results. An independent verification confirmed that the unweighted baseline Random Forest model reproduces identically across scikit-learn versions 1.6.1 and 1.9.0, while the balanced class-weight model does not: under otherwise identical code, data, and random seed, West African recall for the balanced model shifts from 0.294 under scikit-learn 1.6.1 to 0.513 under scikit-learn 1.9.0. No defect was identified in either environment; this is disclosed as a version-dependent methodological limitation rather than an error.

## Data availability

Raw audio previews are not redistributed because of copyright restrictions on commercial streaming content. `data/corpus_metadata.csv` is the corpus metadata manifest; its `preview_path` field records the local file location used during the analysis, `data/raw/audio/<genre>/<track_id>.m4a`. The manifest contains 366 records; two records have no preview audio available and are therefore absent from the 364-track feature table. To reproduce the corpus, run `scripts/collect.py` with valid Spotify credentials, or place preview files with the same naming convention in `data/raw/audio/<genre>/`.

## Reproduction

The pipeline executes in the following order. All outputs are written to `data/processed/`.

1. Corpus retrieval

```bash
python scripts/collect.py
```

Retrieves track metadata and thirty-second preview audio, writing `data/metadata/corpus.jsonl` and `data/raw/audio/<genre>/`. The published manifest, `data/corpus_metadata.csv`, is the CSV form of this file.

2. Feature extraction: `notebooks/02_feature_extraction.ipynb`

Scans `data/raw/audio/`, converts each preview to 22,050 Hz mono, standardizes duration to thirty seconds, and extracts the tonal, temporal, dynamic, and timbral descriptors. The resulting feature table is `data/processed/features.csv`.

3. Exploratory data analysis: `notebooks/03_exploratory_data_analysis.ipynb`

Loads `data/processed/features.csv` and produces `features_eda_clean.csv`, `normality_test_results.csv`, and `feature_effect_sizes.csv`, together with the distributional, correlational, and principal component figures.

4. Statistical comparison: `notebooks/04_statistical_comparison.ipynb`

Loads `features_eda_clean.csv` and produces the test-selection summary, corpus and genre comparison tables, post-hoc Caribbean-versus-West-African results, `acoustic_proximity_ranking.csv`, and the associated figures.

5. Similarity analysis: `notebooks/05_similarity_analysis.ipynb`

Loads `features_eda_clean.csv` and produces `genre_similarity_profile.csv` and `feature_level_similarity.csv`.

6. Clustering and classification: `notebooks/06_clustering_and_classification.ipynb`

Loads `features_eda_clean.csv` and produces `cluster_assignments.csv`, `cluster_evaluation_summary.json`, `table7b_class_metrics_comparison.csv`, `table7b_overall_metrics_comparison.csv`, and `figure10b_confusion_matrix_comparison.png`. The hierarchical clustering dendrogram with genre profiles is rendered in the notebook.

## Citation

Thesis citation: Bardouille, M. (2026). From Port to Playlist: Acoustic Overlap and Timbral Divergence in Caribbean and West African Music [Master's thesis, STRUCTURALIA].

Repository: https://github.com/BardouilleMhea/port_to_playlist/blob/main/README.md
Release tag: `<RELEASE_TAG>`
Zenodo DOI: `<ZENODO_DOI>`

## License

MIT License. See LICENSE.
