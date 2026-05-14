# Reproduction Guide

This document describes the recommended order for reproducing the synthetic benchmark, baseline comparison, FORESEE field-data application, and sensitivity analysis.

## 1. Prepare Environment

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The scripts were developed for a local workstation workflow. Before running, update the hard-coded paths in the scripts or adapt them to your own project directory.

## 2. Generate Synthetic Data

```bash
python scripts/synthetic_data_generator.py
```

Recommended synthetic configuration:

```text
configs/synthetic_config.json
```

The generator produces benchmark artifacts and coherent-signal rescue cases.

## 3. Check Synthetic Dataset

```bash
python scripts/check_synthetic_dataset.py
```

This step checks sample count, required fields, SNR distribution, mask ratio, and basic statistics.

## 4. Calibrate STA/LTA Slicing Parameters

```bash
python scripts/calibrate_synthetic_sta_lta_params.py
```

This step calibrates the candidate-event slicing parameters used by MCS-AND.

## 5. Run Synthetic MCS-AND Benchmark

```bash
python scripts/run_synthetic_mcsand_benchmark.py
```

The benchmark evaluates:

- MCD only;
- MCD + plain semblance;
- MCD + `τ-p`;
- MCD + `τ-p` + local low-rank repair.

## 6. Run Baseline Comparison

```bash
python scripts/run_synthetic_baselines.py
```

The baseline comparison includes:

- bandpass + median filtering;
- global SVD;
- DAS-N2N when available;
- MCS-AND variants.

## 7. Plot Synthetic and Baseline Figures

```bash
python scripts/plot_mcsand_paper_figures.py
python scripts/plot_baseline_comparison_figures.py
python scripts/plot_synthetic_example_figures_a4.py
```

These scripts generate publication-ready figures.

## 8. Prepare FORESEE Field Cases

Download the FORESEE source files listed in:

```text
configs/foresee_case_list.csv
```

Then extract the field subsets:

```bash
python scripts/extract_foresee_real_cases.py
```

The field subset uses channels 1460-1860 and the first 60 s of each selected HDF5 file.

## 9. Run MCS-AND on FORESEE Field Data

```bash
python scripts/run_real_foresee_mcsand.py
python scripts/plot_real_foresee_mcsand_figures_overlay_standalone.py
```

The FORESEE field examples are used to evaluate signal preservation and event-level behavior in real urban DAS records. Because field artifact labels are unavailable, these experiments do not report F1 scores.

## 10. Optional DAS-N2N Field Comparison

```bash
python scripts/prepare_foresee_for_dasn2n.py
python scripts/run_dasn2n_on_foresee_cases.py
python scripts/plot_real_dasn2n_mcsand_comparison.py
```

This comparison evaluates waveform preservation and residual structure rather than supervised accuracy.

## 11. Sensitivity and Robustness Analysis

```bash
python scripts/run_plot_mcsand_sensitivity_robustness.py
```

This generates:

- final-noise-count sensitivity curves;
- alpha-outlier-fraction calibration curves.

## 12. Notes on Data Availability

Large data files and generated outputs are not tracked by Git. They should be regenerated locally or archived as release assets on Zenodo.
