# Reproducing the denoising results

### Requirement:
- conda installed and accessible in a command prompt

### Procedure:
1. create a conda environment from the provided `environment.yml`: `conda env create -f environment.yml`.
2. Download the datasets `dexter.csv`, `music_train.csv`, `music_test.csv`, `wdc_almser_test.csv`, `wdc_almser_train.csv`.
3. open a command prompt (make sure conda is installed and accessible there) and run `run_pipeline.bat`.
4. After the pipeline has finished, go into src/evaluation to identify the evaluation results you want to reproduce.