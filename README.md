# Contour Alternatives

Software for generating alternative segmentation contours based on clustering the total discrepancy between two segmentations into spatially and directionally coherent regions.
This is the first release of the research code. For any questions or comments please contact :)

Email: j.e.van.aalst@umcg.nl | joelle.vanaalst@live.nl

## Data expectation

The workflow expects the data (CT, true labels, predicted labels) as NIfTI files in three directories:

```text
imagesTs/
	HNC-B_001_0000.nii.gz
labelsTs_true/
	HNC-B_001.nii.gz
labelsTs_pred/
	HNC-B_001.nii.gz
```

The CT filename must end in `_0000.nii` or `_0000.nii.gz`. The ground-truth and prediction files use the same patient stem without `_0000`. All three arrays must have the same shape. Prediction and ground-truth values are interpreted as integer labels, with `0` as background.

## How to get the code to run?

You can specify the paths (and other variables) as a config file. You can copy
[configs/generation.example.json](configs/generation.example.json), edit its paths (and other variables), and run the two scripts in order:

```powershell
python optimise_clustering.py --config configs/[your config file].json
python generate_contour_alternatives_final.py --config configs/[your config file].json
```

You can also call the functions directly from a small Python file:

```python
from contour_alternatives.config import GenerationConfig
from contour_alternatives.workflows.optimise import optimise_clustering
from contour_alternatives.workflows.generate import generate_contour_alternatives

config = GenerationConfig.from_json("configs/generation.local.json")
optimise_clustering(config)
generate_contour_alternatives(config)
```

## Quick start

To use the files you need to download the folder, open a terminal in that folder, and install the dependencies (if not already installed):

```powershell
python -m pip install -r requirements.txt
```

This installs the dependencies for the generation and clustering-optimisation workflows.

Then you can run the workflows in order:

```powershell
python optimise_clustering.py --config configs/my_config.json
python generate_contour_alternatives_final.py --config configs/my_config.json
```

## Functionality of the script
The script has two functions, executed in two steps:
1. Optimising the hyperparameter tuning for the clustering algorithm (using Optuna)
2. Executing the clustering to generate contour alternatives

### Step 1: Optimise clustering

Run [optimise_clustering.py](optimise_clustering.py). It calls the function `optimise_clustering(config)` from [workflows/optimise.py](src/contour_alternatives/workflows/optimise.py). It tests different clustering parameters separately for each patient and OAR, then writes `custom_tuning_results.csv` with the hyperparameter tuning results.

The clustering workflow:
1. Calculate edit vectors between the predicted and ground-truth surfaces for each patient and OAR.
2. Remove edit vectors shorter than `edit_threshold`.
3. Compute and scale the spatial and edit features used for clustering.
4. Use Optuna to test different hyperparameter combinations for each patient/OAR pair.
    a. Run HDBSCAN for each trial and labels points as clusters or noise.
    b. Score each trial using `silhouette`, `db`, or `custom` scoring.
    c. Repeat the trial process, with 50 trials by default.
5. Write all trial results to `custom_tuning_results.csv` using semicolon separators.

The next workflow uses this CSV to select the highest-scoring parameter combination separately for each patient and OAR. At present, the optimisation implementation uses HDBSCAN for the trials; support for tuning DBSCAN separately can be added later if needed.

#### Hyperparameters

Optuna searches for the optimal parameters for:
| Parameter | Meaning |
| --- | --- |
| `weight_spatial` | Weight applied to the spatial origin features |
| `weight_radial_distance` | Weight applied to radial edit distance |
| `weight_angle` | Weight applied to the angular edit features |
| `min_samples` | Minimum number of samples required by HDBSCAN to identify a dense region. |

The following values are configured rather than optimised in the current implementation:
| Parameter | Meaning |
| --- | --- |
| `min_cluster_size` | fixed at `2`. |
| `n_trials` | number of Optuna trials, default `50`. |
| `optimisation_goal` | `silhouette`, `db`, or `custom`. |
| `alpha` | weighting of spatial coherence versus edit-feature coherence in the `custom` score, default `0.8`. |

#### Custom scoring objective

`custom` is the advised `optimisation_goal`, as used and explained in the accompanying paper (see Reference). When set, each trial is scored as:

```text
maximise ( alpha * S_cart_bar + (1 - alpha) * S_sph_bar - lambda * max(0, K - 30) )
```

- `S_cart_bar` spatial coherence: silhouette score on each edit vector's origin `(px, py, pz)`, rescaled to [0,1]. Are edits in a cluster from the same anatomical area?
- `S_sph_bar` directional coherence: mean intra-cluster distance on each edit vector's `(r, theta, phi)`, rescaled to [0,1]. Do edits in a cluster point the same way, by a similar amount?
- `K` number of clusters; `lambda = 0.1` penalises trials with `K > 30` to discourage over-fragmentation.
- `alpha` fixed weight balancing the two coherence terms (see below).

**`alpha`** controls how much spatial vs. directional coherence matters: `1` = location only, `0` = direction only. The default `0.8` favours location, so edits are grouped mainly by proximity, with direction as a secondary criterion.


### Step 2: Generate alternatives

Run [generate_contour_alternatives_final.py](generate_contour_alternatives_final.py). It calls `generate_contour_alternatives(config)` from [workflows/generate.py](src/contour_alternatives/workflows/generate.py). When `use_tuned_hyperparameters` is `true`, it reads the tuning CSV and selects the highest-scoring settings for each patient/OAR pair. When it is `false`, it uses the defaults in the workflow.

The generation workflow:

1. Compute edit vectors and weighted clustering features.
2. Run HDBSCAN or DBSCAN using the optimised or default hyperparameters
3. Expand clusters into surface regions.
4. Generate the alternative contour per cluster.
5. Write the alternative and changed-region NIfTI files.
6. Writes `Statistics.xlsx` in the alternatives directory with the robustness and geometry evaluation

## Config
Both scripts expect a config file, with the following fields:

| Field | Meaning |
| --- | --- |
| `paths.ct` | CT NIfTI directory |
| `paths.ground_truth` | Ground-truth segmentation directory |
| `paths.prediction` | Prediction directory |
| `paths.alternatives` | Output directory |
| `paths.tuning_results` | Semicolon-separated tuning CSV |
| `paths.tuning_output` | Location where Step 1 writes the tuning CSV |
| `margin` | Bounding-box margin in voxels |
| `dilation_size` | Surface-region dilation size |
| `edit_threshold` | Minimum edit length used for clustering |
| `clustering_algorithm` | `HDBSCAN` or `DBSCAN` |
| `use_tuned_hyperparameters` | Select the highest-scoring tuning row when available |
| `dbscan_eps` | DBSCAN neighbourhood radius |

Set `use_tuned_hyperparameters` to `false` and remove `paths.tuning_results` if you want to use defaults only. During the two-step workflow, `paths.tuning_output` and `paths.tuning_results` should point to the same file.


## Outputs

For a patient/OAR/cluster combination, the workflow writes:

```text
<patient>_<oar>_<cluster>.nii.gz
<patient>_<oar>_<cluster>_changed.nii.gz
```

It also writes `Statistics.xlsx` with spatial cluster results, spatial region results, volume results, and geometric accuracy results. 

## Known limitations
- The code now also needs a CT, this was necessary for other functionality that this script was used for and is (technically) not necessary for this script. Though it cannot be left out. To do: fix possibility of running without CT available

## Reference

van Aalst, J. et al. *Which Edits Matter? Simulating Realistic Local Corrections to Organ-of-Interest DL Segmentation and Predicting Dosimetric Impact.* MIART 2026 (submission).
