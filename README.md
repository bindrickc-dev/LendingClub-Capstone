# LendingClub Capstone: Week 1–11 Notebooks

## Setup

1. Download `accepted_2007_to_2018Q4.csv.gz` from https://www.kaggle.com/datasets/wordsforthewise/lending-club and put it at `data/accepted_2007_to_2018Q4.csv.gz`. Don't unzip it; pandas reads `.gz` directly.
2. Install the packages: `pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pyarrow jupyter`. Use scikit-learn 1.2 or newer.
3. Run the notebooks **in order**. Week 1 builds `data/lc_model_sample.parquet`, the shared sample and train/test split. Week 4 saves the test-set scores that Weeks 5, 6, 8 and 9 compare against. Run Week 10 before Week 11.

| Notebook | Concepts | Target |
|---|---|---|
| Week 1 | data prep, target definition, EDA, OLS with continuous and categorical features, polynomial and interaction terms, multicollinearity / VIF | `int_rate` |
| Week 2 | ridge, lasso, elastic net (α chosen by CV), coefficient paths | `int_rate` |
| Week 3 | forward and backward selection, PCR, PLSR | `int_rate` |
| Week 4 | logistic regression, feature scaling, tuning C and class weight, threshold, calibration, odds ratios | charge-off |
| Week 5 | linear and RBF SVM, the kernel trick, C and γ regularization, comparison with logistic regression | charge-off |
| Week 6 | decision tree, random forest, OOB, feature importance, final model comparison | charge-off |
| Week 8 | KNN, k and distance metrics (Euclidean, Manhattan, cosine), scaling, curse of dimensionality | charge-off |
| Week 9 | gradient boosting: learning rate, number of trees, depth, regularization, early stopping, permutation importance | charge-off |
| Week 10 | K-Means, feature scaling, elbow method, silhouette score, stability, cluster profiles vs. grade | none (unsupervised) |
| Week 11 | DBSCAN (k-distance plot, noise), hierarchical clustering, linkage methods, dendrogram | none (unsupervised) |

## Output folders

- `figures/` — figures for the reports: `fig1`–`fig6` match Figures 1–6 in the Milestone One report; `fig_gb_overfitting_tuning`, `fig_kmeans_elbow_silhouette` and `fig_kmeans_profiles_k4` are Figures 1–3 in Milestone Two (`fig_dbscan_hac` is the Week 11 figure).
- `results/` — CSV and parquet files of the metrics and test-set scores.

## Runtime

Approximate. It depends on cores and memory; the full CSV needs about 4–6 GB of RAM.

- Week 1 takes a few minutes. Most of that is the full-file missingness scan; set `RUN_FULL_MISSINGNESS_SCAN = False` to skip it.
- Week 3 runs hundreds of fits for stepwise selection and PCR. Expect 10–20 minutes, or lower `SELECTION_N`.
- Week 5's RBF SVM is tuned on 10k rows and refit on 20k (`SVM_TUNE_N`, `SVM_FIT_N`).
- Week 6's random-forest search uses a 50k subsample (`RF_TUNE_N`).
- Week 8 tunes KNN on a 30k subsample (`KNN_N`), about 4 minutes. Week 9 takes about 10 minutes (randomized search). Weeks 10–11 take a few minutes; DBSCAN and HAC use 20k and 5k samples (`DB_N`, `HAC_N`).

To make every notebook run faster, lower `N_MODEL` in Week 1 (default 200,000), then rerun Weeks 2–11.

## Conclusions and sources

Each notebook ends with written conclusions drawn from its own outputs, followed by the high-quality source used for that week's topic (also shared on Yellowdig). The Milestone One and Two reports summarize these results; every number in them comes from these saved outputs.
