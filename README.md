# petm_mixing_model summary:

Three end-member mixing model for Inglis et al. Nature Geoscience 2026. This mixing model allows for the fact that some conservative tracers are insensitive to some potential organic carbon end-members (e.g., BIT does not track plant inputs whereas TAR does not track soil inputs).

---

# Repository information:

| Field   | Value                         |
| ------- | ----------------------------- |
| Author  | Jordon D. Hemingway           |
| Date    | 17. April 2026                |
| Contact | jordon.hemingway@eaps.ethz.ch |
| License | GNU GPL v3                    |
| doi     | [![DOI](https://zenodo.org/badge/1213451443.svg)](https://doi.org/10.5281/zenodo.19630720) |


---

# Repository architecture:

## 01 analysis code
Contains two .py files, ``mixfuncs.py`` and ``petm_unmixing.py``. The former contains all functions necessary to performing mixing model. The latter contains the scripts to run the mixing model exactly as done for Inglis et al. 2026.

## 02 input data
Contains all input data for all PETM cores used in Inglis et al. 2026.

## 03 output data
Contains all model-generated output data, including Monte Carlo uncertainty on fractional end-member contributions.

## 04 output figures
Contains all model-generated output figures.

## 05 documentation
Contains all theory and equations used for model development.

---
