# GitHub and Zenodo Release Guide

This guide describes how to publish the repository and generate a citable DOI.

## 1. Create a GitHub Repository

Recommended repository name:

```text
MCS-AND-DAS-artifact-suppression
```

Recommended repository description:

```text
Code, synthetic benchmarks, and FORESEE field-data workflows for MCS-AND: training-free artifact suppression in distributed acoustic sensing.
```

Recommended settings:

```text
Visibility: Private during manuscript preparation, Public after final checking
.gitignore: Python
License: MIT
```

## 2. Upload the Local Repository

Option A: GitHub Desktop

1. Open GitHub Desktop.
2. Choose `File` -> `Add local repository`.
3. Select this folder.
4. Commit all files.
5. Publish the repository to GitHub.

Option B: Command Line

```bash
git init
git add README.md LICENSE CITATION.cff requirements.txt .gitignore configs docs models tables scripts
git commit -m "Initial release of MCS-AND DAS artifact suppression workflow"
git branch -M main
git remote add origin https://github.com/<your-user-name>/MCS-AND-DAS-artifact-suppression.git
git push -u origin main
```

## 3. Check Before Public Release

Before making the repository public, check that no large or private files are included:

```bash
git status
git ls-files
```

Do not include:

```text
*.hdf5
*.npz
*.npy
raw DAS data
generated benchmark folders
private local path files
```

## 4. Create a Versioned Release

After final checking, create a GitHub release:

```text
Tag: v1.0.0
Title: MCS-AND DAS artifact suppression workflow v1.0.0
```

Attach optional release assets if needed:

```text
synthetic_mcd_model.pkl
final CSV result tables
small metadata files
```

Large data should be archived on Zenodo rather than tracked by Git.

## 5. Generate a Zenodo DOI

1. Log in to Zenodo.
2. Go to GitHub integration.
3. Authorize Zenodo to access your GitHub account.
4. Enable archiving for this repository.
5. Create a GitHub release, for example `v1.0.0`.
6. Zenodo will automatically archive the release and generate a DOI.
7. Replace the DOI placeholder in `README.md` and `CITATION.cff`.

## 6. Suggested Data Availability Statement

```text
The processing scripts, parameter configurations, synthetic artifact generator, MCD model description, and FORESEE subset manifest are available in the GitHub repository and archived on Zenodo. The synthetic benchmark can be regenerated using the provided scripts. The FORESEE field records are from the public PubDAS repository; only source filenames, channel ranges, time windows, and preprocessing metadata are provided here. The original field data are not redistributed.
```
