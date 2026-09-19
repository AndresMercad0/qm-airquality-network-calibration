# Reproducing the tables and figures

Every number in the paper comes from a CSV under `results/`. This page lists which script writes each CSV and which table or figure reads it. Commands are run from the repository root. `N` stands for the node (3 or 5).

## Order of execution

| Order | Command | Needs | Writes |
|---|---|---|---|
| 1 | `python pipeline/run_pipeline.py --node N --skip s9` | dataset clone | `data/node_N/`, `results/node_N/s5_v1v2/`, `s6_v3/`, `s7_evaluate/` |
| 2 | `python pipeline/run_pipeline.py --node N --from s9` | step 1, GPU recommended | `results/node_N/s9_ml/tabla_maestra_resultados.csv`, `metricas_drift_mensual.csv`, `models/` |
| 3 | `python pipeline/s9_train_no2alpha_only.py --node N` and `python pipeline/s9_train_no2alpha_lstm_only.py --node N` | step 1 | `tabla_maestra_no2alpha.csv`, `tabla_maestra_no2alpha_lstm.csv` |
| 4 | `python experiments/s9f_lstm_search_v6.py --node N` | steps 1 to 3, GPU recommended | `tabla_maestra_lstm_v6.csv`, `lstm_v6_cv/` |
| 5 | `python experiments/s9b_locked_selection.py --node N` | steps 1 to 4 | `results/cross_node/locked_selection_*_nodeN.csv` |
| 6 | `python experiments/s9c_single_basis_evaluation.py --node N` | steps 1 to 4 | `results/cross_node/single_basis_metrics_nodeN.csv` |
| 7 | `python experiments/s9d_ablation.py --node N` | steps 1 and 2 | `tabla_maestra_ablacion_v6.csv` |
| 8 | `python experiments/s9e_alphasense_no2_monthly.py --node N` | step 1 | `metricas_alphasense_no2_ols_mensual_v6.csv` |
| 9 | `python experiments/s9h_ols_validation.py --node N` | steps 5 and 6 | `ols_validation_nodeN.csv`, `method_selection_v6.csv` |
| 10 | `python experiments/s9i_ols_transfer.py` | step 1, both nodes | `ols_transfer_v6.csv` |
| 11 | `python experiments/s9j_ols_ml_monthly.py --node N` | steps 5 and 6 | `metricas_ols_ml_mensual_v6.csv` |
| 12 | `python experiments/s9g_build_v6_figure_inputs.py --node N` | steps 5 and 6 | `metricas_por_nivel_de_costo_v6.csv`, `tabla_maestra_locked_v6.csv`, `tabla_maestra_v6.csv` |
| 13 | `python experiments/s10d_source_selected_transfer.py` | steps 1 to 5, both nodes | `source_selected_transfer.csv` |
| 14 | `python experiments/build_fig11_input.py` | step 13 | `best_transfer_source_selected_v6.csv` |

Steps 1, 8 and 10 need no trained model. Every other experiment loads the models written by steps 2 to 4 into `results/node_N/s9_ml/models/`.

## Tables

| Table | Content | Source |
|---|---|---|
| 5 | Validation-selected model per sensor, test R2 and validation RMSE | `locked_selection_per_gas_nodeN.csv`, `tabla_maestra_v6.csv` |
| 6 | Ablation of the inputs (Random Forest) | `tabla_maestra_ablacion_v6.csv` |
| 7 | Transfer between the two nodes, scored on the destination test period (`period = test`) | `source_selected_transfer.csv` |
| A1 | Inputs of each configuration | `pipeline/config.py` (`ML_FEATURES_*`) |
| A2 | Hyperparameter search spaces | `pipeline/config.py` (`HP_*`), `pipeline/ml/models.py` |
| A3 | Observations per target | `tabla_maestra_v6.csv` (`n_train`, `n_test`) |
| A4 | The two selection rules | `locked_selection_per_gas_nodeN.csv` |
| A5 | Validation RMSE of OLS and of the selected model | `ols_validation_nodeN.csv`, `method_selection_v6.csv` |
| A6 | Alphasense calibration constants | `pipeline/config_node3.py`, `pipeline/config_node5.py` |

## Figures

Each figure has a script and a JSON configuration in `figures/`; the PNG is written to `figures/output/`.

| Figure | Script | Reads |
|---|---|---|
| 1 | `fig01_sites_overview.py` | site photograph, map tiles (see the note below) |
| 2 | `fig02_pipeline_flowchart.py` | configuration only |
| 3 | `fig03_r0_dynamic.py` | `data/node_3/s4_preprocess/` (written by step 1) |
| 4 | `fig04_temporal_design.py` | configuration only |
| 5 | `fig05_cross_node_protocol.py` | configuration only |
| 6 | `fig06_r2_comparison.py` | `metricas_por_nivel_de_costo_v6.csv`, `tabla_maestra_locked_v6.csv` (both nodes) |
| 7 | `fig07_no2_three_tier.py` | `single_basis_metrics_node3.csv` |
| 8 | `fig08_obs_vs_pred_density.py` | `data/node_3/`, `tabla_maestra_v6.csv` and the trained models of node_3 |
| 9 | `fig09_ml_landscape.py` | `tabla_maestra_v6.csv` (both nodes) |
| 10 | `fig10_monthly_r2.py` | `metricas_ols_ml_mensual_v6.csv`, `metricas_drift_mensual.csv`, `metricas_alphasense_no2_ols_mensual_v6.csv` (node_5) |
| 11 | `fig11_cross_node_transfer.py` | `best_transfer_source_selected_v6.csv` |
| 12 | `fig12_cost_performance.py` | configuration (values from `method_selection_v6.csv`) |

Table 7 reads the rows of `source_selected_transfer.csv` with `period = test`. Direct transfer: `selection_rule = locked_min_val_RMSE` and `ols_variant = source` (`none` for Configuration A, which has no OLS input). OLS-refitted: the same model with `ols_variant = destination_train`. Oracle: the highest destination R2 among the candidate models (the three selection rules) with `ols_variant = destination_train` or `none`. The candidate rule `v5_parsimony` is the simplest family within 0.05 of the highest test R2. Rows with `ols_variant = destination_full` or `period = full` are not reported in the paper.

Figure 1 draws its basemap from CARTO tiles through `contextily`. At the time of release the tile service returned tiles watermarked "API KEY REQUIRED", so the map panel does not match the published figure.

## Stages of Figure 2 and scripts

Figure 2 of the paper numbers the stages in their logical order. The scripts keep the numbers under which they were developed, so the two numberings differ from stage s5 onwards.

| Stage in Figure 2 | Script |
|---|---|
| s0 Download raw data | the raw data are published in `qm-airquality-network-data`; `pipeline/s0_prepare_inputs.py` crops them to the study window |
| s1 to s4 | `pipeline/s1_qc_lowcost.py`, `s2_convert_alphasense.py`, `s3_qc_alphasense.py`, `s4_preprocess.py` |
| s5 Assemble final dataset | `pipeline/s8_assemble_final.py` |
| s6 Calibrate: manufacturer curves | `pipeline/s5_calibrate_v1v2.py` |
| s7 Calibrate: OLS | `pipeline/s6_calibrate_v3.py` |
| s8 Calibrate: machine learning | `pipeline/s9_calibrate_ml.py` and the two `s9_train_no2alpha*` scripts |
| s9 Evaluate all methods | `pipeline/s7_evaluate.py` and `experiments/s9c_single_basis_evaluation.py` |
| s10 Cross-node transfer validation | `experiments/s10d_source_selected_transfer.py` |

## Numerical agreement

Steps s0 to s8 are deterministic: run on the published dataset and the reference snapshot, they reproduce the CSVs of `results/node_N/s5_v1v2`, `s6_v3` and `s7_evaluate` exactly. Scoring the stored models on another machine reproduces the test R2 of the master tables within 0.0003.
