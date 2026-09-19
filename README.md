<!-- title: QM Air Quality Network Calibration -->
# QM Air Quality Network Calibration

Calibration pipeline and analysis code for the low-cost and mid-cost sensors of the **QMUL Air Quality Monitoring System (AQMS)**. Two outdoor nodes (node_3 at Queen Mary University of London, Mile End Road, and node_5 at King Edward Memorial Park, London) are calibrated against the co-located Tower Hamlets reference stations TH2 and TH7 over ten months (16 April 2025 to 15 February 2026), with three methods: manufacturer curves, ordinary least squares by co-location, and machine learning.

The measurements are published as a companion dataset, **[qm-airquality-network-data](https://github.com/AndresMercad0/qm-airquality-network-data)**, and the stations themselves (hardware design and firmware) are documented at **[qm-airquality-network](https://github.com/AndresMercad0/qm-airquality-network)**.

## What this repository contains

- `pipeline/` - the calibration pipeline, steps s0 to s9. One codebase for both nodes; the node is chosen with `--node 3` or `--node 5`.
- `experiments/` - the analyses built on top of the pipeline: selection of the machine-learning model on a validation block, evaluation of every method on one common set of test observations, ablation of the inputs, monthly series and transfer between the two nodes.
- `figures/` - one script and one JSON configuration per figure of the paper.
- `results/` - the result tables behind every table and figure of the paper (CSV, about 1 MB).
- `data/reference_snapshot/` - the two reference-station files used by the study (see [Reference data attribution](#reference-data-attribution)).
- `docs/REPRODUCING.md` - which script and which CSV produce each table and figure.

The trained models are not stored here because of their size (4.3 GB).

## Requirements

Python 3.12 and the packages in `requirements.txt` (the versions used for the published results). The figure scripts also need `requirements-figures.txt` and a local Chrome or Chromium, which renders the HTML figures to PNG; set `AQMS_CHROME` if the browser is not in a standard location.

```
pip install -r requirements.txt
```

The machine-learning step (s9) was run on an NVIDIA GPU; every other step runs on a laptop. On macOS, set `OMP_NUM_THREADS=1` for the scripts that load LightGBM and PyTorch models in the same process; without it the two OpenMP runtimes conflict and the interpreter crashes.

## Reproducing the results

1. Clone the dataset next to this repository, with Git LFS installed, or point `AQMS_DATASET_DIR` to an existing clone:

   ```
   git lfs install
   git clone https://github.com/AndresMercad0/qm-airquality-network-data
   ```

2. Run the pipeline for each node. Steps s0 to s8 take about a minute and a half per node:

   ```
   python pipeline/run_pipeline.py --node 3 --skip s9
   python pipeline/run_pipeline.py --node 5 --skip s9
   ```

   Step s0 crops the published node file to the study window and writes it to `data/node_N/raw/`. The intermediate files of each step are written under `data/node_N/` (not tracked) and the result tables under `results/node_N/`.

3. Train the models (`--from s9`), then run the scripts of `experiments/` and `figures/` in the order given in [docs/REPRODUCING.md](docs/REPRODUCING.md).

The CSVs in `results/` are the ones behind the paper, so any table or figure can be checked without retraining.

## Pipeline, at a glance

| Step | Script | What it does |
|---|---|---|
| s0 | `s0_prepare_inputs.py` | Crops the published node file to the study window |
| s1 | `s1_qc_lowcost.py` | Quality control of the low-cost signals: ADC saturation, stuck values, Hampel filter, physical bounds |
| s2 | `s2_convert_alphasense.py` | Alphasense voltages to ug/m3 (application note AAN 803-05) |
| s3 | `s3_qc_alphasense.py` | Quality control of the Alphasense series: bounds, Hampel filter, thermal transients |
| s4 | `s4_preprocess.py` | Metal oxide signal to sensor resistance, temperature and humidity correction, dynamic R0, ratio |
| s5 | `s5_calibrate_v1v2.py` | Calibration with the manufacturer curves |
| s6 | `s6_calibrate_v3.py` | Calibration by ordinary least squares against the co-located reference or comparator |
| s7 | `s7_evaluate.py` | Metrics per pollutant and sensor tier |
| s8 | `s8_assemble_final.py` | Assembles the calibrated series |
| s9 | `s9_calibrate_ml.py` | Machine-learning calibration: five targets, three input configurations, five model families, random search with time-series cross-validation |

`cross_node_features.py` is not a step: it builds the feature table of the partner node for the transfer experiment of `experiments/`.

Low-cost sensors: Winsen GM-702B (CO), GM-102B (NO2) and MQ131 (O3), and Sensirion SPS30 (PM2.5). Mid-cost comparators: Alphasense NO2-B43F, OX-B431 and CO-B4. NO2 and PM2.5 are evaluated against the reference stations; CO and O3 against the co-located Alphasense cells, because TH2 and TH7 provide only NO2 and PM2.5.

## Per-node differences

Both nodes run the same code. What changes per node is in `pipeline/config_node3.py` and `pipeline/config_node5.py`: the reference station (TH2 or TH7), the calibration sheet of the three Alphasense cells and the partner node used in the transfer step.

## Reference data attribution

Government reference data from stations TH2 (Mile End Road) and TH7 (King Edward Memorial Park) were obtained from the Environmental Research Group of Imperial College London, using data from the London Air Quality Network. This information is licensed under the terms of the **Open Government Licence v2.0**. Please give the same credit when reusing the files under `data/reference_snapshot/`.

The snapshot holds the reference records as downloaded on 15 February 2026, which are the ones used in the study. The reference files of the dataset repository were downloaded on 10 June 2026 and their NO2 values differ from the snapshot, so the snapshot is needed to reproduce the published numbers.

## Related outputs

- Dataset: [qm-airquality-network-data](https://github.com/AndresMercad0/qm-airquality-network-data)
- Hardware and firmware: [qm-airquality-network](https://github.com/AndresMercad0/qm-airquality-network)
- Network design paper: Mercado-Velazquez, A. A., Poslad, S., & Escamilla-Ambrosio, P. J. (2026). *Design and Implementation of a Low-Cost IoT Air-Quality Monitoring Network at Queen Mary University of London Mile End Campus*. Smart Cities (ICSC-CITIES 2025), CCIS vol. 2742, Springer. https://doi.org/10.1007/978-3-032-19019-2_7
- Calibration paper: manuscript under review.

## Licence and citation

Released under the [MIT Licence](LICENSE). If you use this code in academic work, please cite it (see [CITATION.cff](CITATION.cff)) and the papers above.

Release v1.0.0 is archived on Zenodo: https://doi.org/10.5281/zenodo.22850320. The DOI https://doi.org/10.5281/zenodo.22850319 resolves to the latest archived version.

## Contact

Andres Aharhel Mercado-Velazquez, IoT2US Lab, School of Electronic Engineering and Computer Science, Queen Mary University of London. a.mercadovelazquez@qmul.ac.uk
