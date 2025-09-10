# Reproducing the Denoising Results

## Requirements

Before running the pipeline, make sure the following are installed and available in your system’s command prompt:

* [Conda](https://docs.conda.io/en/latest/miniconda.html)
* [R](https://cran.r-project.org/) (with access to `Rscript.exe`)

## Procedure

1. **Set up the environment**
   Create a conda environment using the provided file:

   ```bash
   conda env create -f environment.yml
   ```

2. **Install the following R packages**
   * `dplyr`
   * `glue`
   * `jsonlite`
   * `NoiseFiltersR` (see [CRAN](https://cran.r-project.org/web/packages/NoiseFiltersR/index.html))

3. **Obtain the datasets**
   Download the following files and place them in the appropriate folder in the `dataset` directory:

   * `dexter.csv`
   * `music_train.csv`
   * `music_test.csv`
   * `wdc_almser_train.csv`
   * `wdc_almser_test.csv`

4. **Configure R path**
   Open `run_pipeline.bat` in a text editor and update the `RPATH` variable so it points to the correct `Rscript.exe` on your system.

5. **Run the pipeline**
   Open a command prompt (with Conda accessible) and execute:

   ```bat
   run_pipeline.bat
   ```

6. **Inspect results**
   Once the pipeline completes, navigate to `src/evaluation` to find and review the evaluation results.
