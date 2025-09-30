@echo off

set dataset=music
set RPATH="C:\Program Files\R\R-4.3.1\bin\Rscript.exe"

%RPATH% %SCRIPT% %DATASET%

call remove_all_temp_files.bat
call run_all_tests.bat

call conda activate master-thesis-env

cd src

echo ################### PREPROCESSING ###################
python 1_preprocessing.py

echo ################### NOISE INTRODUCTION ###################
python 2_introduce_artificial_noise.py %dataset%

echo ################### DENOISING ###################
python 3_denoising_evaluation.py %dataset%
%RPATH% ".\r_programming\3_denoise_evaluation_scenarios.R" %dataset%

echo ################### DOWNSTREAM CLASSIFICATION ###################
python 4_downstream_performance_evaluation.py train_test_split %dataset%
python 4_downstream_performance_evaluation.py cleaning %dataset%
%RPATH% ".\r_programming\3_denoise_evaluation_scenarios_downstream_classifier.R" %dataset%
python 4_downstream_performance_evaluation.py evaluation %dataset%

pause
