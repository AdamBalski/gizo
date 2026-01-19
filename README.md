# GIZO - white blood cells detection, classification and annotation
## Dataset
- Kaggle Blood Cells dataset: https://www.kaggle.com/datasets/paultimothymooney/blood-cells
- `make download` fetches and unpacks the data into `data/`.

## Approach
1. Detect coarse blue-dominant masks on TRAIN images, run connected components, and keep components with adequate area/solidity.
2. For each component, extract a 27-dimensional feature vector capturing bounding-box ratios, color fractions, select RGB statistics, key mask component counts/holes (nucleus/pale/blue), compactness, and nucleus moment metrics.
3. Train a purity-aware ID3 decision tree on TRAIN features.
4. Apply the trained tree to every TEST image: run the same mask + component pipeline, classify each component, draw boxes/labels, and log aggregate stats.
5. Evaluate accuracy over the TEST directory, save detailed results, and generate performance plots plus tree visualizations for inspection.

## Make Targets
- `download` – fetch and unzip the dataset (requires Kaggle credentials).
- `clean-data` – remove the downloaded dataset.
- `clean` – remove all artifacts under `target/`.
- `create-bounding-boxes` – run `bounding_boxes_creation.py` to produce `target/bounding-boxes.json` from TRAIN images.
- `extract-features` – run `feature_extraction.py` to produce `target/features.json`.
- `eval` – train/evaluate the decision tree and write `target/eval-results.json`.
- `summary` – render `target/eval-summary.png` from the evaluation report.
- `render-decision-tree` – render `target/decision-tree.png` from the trained tree.

### Annotating the TEST set
`python3 decision_tree_annotator.py` trains the purity-aware ID3 tree on TRAIN features and automatically annotates every image under `data/dataset2-master/dataset2-master/images/TEST`. Annotated outputs are mirrored under `target/` (matching the dataset tree), and each run prints per-image detection summaries (e.g., `Annotated target/.../TEST/EOSINOPHIL/_0_1022.jpeg :: EOSINOPHIL:1`).

All pipelines run with 12 worker processes and emit simple `current/total` progress counters.
