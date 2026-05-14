# Models

This directory documents model files used by the MCS-AND workflow.

The pretrained synthetic MCD model is not tracked directly in Git because it is a binary output file. The archived release should provide either:

- `synthetic_mcd_model.pkl` as a Zenodo asset, or
- instructions for regenerating it by running the synthetic benchmark workflow.

The MCD model is fitted from the synthetic candidate-event feature library using Yeo-Johnson feature transformation and robust covariance estimation.
