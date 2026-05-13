# MCS-AND-DAS-artifact-suppression
Training-free event-level artifact detection and signal-preserving repair for distributed acoustic sensing (DAS) records.
## Overview

This repository contains the code and configuration files used for the MCS-AND method, a training-free workflow for detecting transient artifacts in DAS records and preserving coherent wavefield events through a τ-p coherence rescue step.

MCS-AND is designed for:

- event-level transient artifact detection;
- robust MCD-based anomaly scoring;
- τ-p coherence-based signal rescue;
- local low-rank repair of final artifact regions;
- reproducible synthetic benchmarking and FORESEE field-data evaluation.

## Repository Contents

```text
configs/     Parameter files and FORESEE subset manifest
scripts/     Data generation, MCS-AND processing, baseline comparison, and plotting scripts
docs/        Reproduction notes
models/      Model description and links to archived pretrained files
tables/      Description of generated result tables
Large raw data files, generated .npz files, benchmark outputs, and figure outputs are not tracked in this GitHub repository.
Data
Synthetic data
The synthetic benchmark can be regenerated using the provided synthetic data generator.

The synthetic data include five artifact scenarios:

spike artifacts;
noncoherent burst artifacts;
moving artifacts;
narrowband artifacts;
hard cases with coherent signal-like structures.
FORESEE field data
The field-data examples use the public FORESEE DAS dataset from PubDAS.

The subset used in this study contains the following source files:

FORESEE_UTC_20190501_120043.hdf5
FORESEE_UTC_20190501_121043.hdf5
FORESEE_UTC_20190501_175043.hdf5
FORESEE_UTC_20190501_180043.hdf5
Subset configuration:

Channel range: 1460-1860
Time window: first 60 s of each file
Sampling rate: 125 Hz
Channel spacing: 2 m
Gauge length: 10 m
The original FORESEE data are not redistributed in this repository. Users should download them from PubDAS and place them in the expected local data directory.

Reproduction
A typical reproduction workflow is:

python scripts/synthetic_data_generator.py
python scripts/check_synthetic_dataset.py
python scripts/calibrate_synthetic_sta_lta_params.py
python scripts/run_synthetic_mcsand_benchmark.py
python scripts/run_synthetic_baselines.py
python scripts/plot_mcsand_paper_figures.py
python scripts/plot_baseline_comparison_figures.py
python scripts/extract_foresee_real_cases.py
python scripts/run_real_foresee_mcsand.py
python scripts/plot_real_foresee_mcsand_figures_overlay_standalone.py
python scripts/run_plot_mcsand_sensitivity_robustness.py
Detailed reproduction instructions are provided in:

docs/reproduce.md
Environment
The code was tested with Python 3.x. Core dependencies include:

numpy
scipy
pandas
matplotlib
scikit-learn
h5py
Install dependencies with:

pip install -r requirements.txt
Outputs
The scripts generate:

synthetic DAS benchmark datasets;
MCS-AND event logs and masks;
baseline comparison tables;
FORESEE field-data application results;
sensitivity and robustness analysis figures;
publication-ready PNG/PDF figures.
Citation
If you use this repository, please cite the associated paper:

[Paper citation to be added after publication]
The archived version of this repository will be available through Zenodo:

Zenodo DOI: 
License
This repository is released under the MIT License.

Contact
For questions about the code or reproduction workflow, please contact:

