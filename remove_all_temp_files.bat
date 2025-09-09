@echo off
setlocal enabledelayedexpansion

:: Define the whitelist of filenames to keep (no paths)
set whitelist=wdc_almser_test.csv wdc_almser_train.csv music_test.csv music_train.csv dexter.csv music_sources_lookup.csv music_raw_features.csv llm_labeled_music_dataset.csv temp_llm_responses.json

:: Define the list of target directories (space-separated)
set target_dirs="C:/Users/Mark/Documents/label_denoising_record_linkage/datasets/wdc_almser" "C:/Users/Mark/Documents/label_denoising_record_linkage/datasets/music" "C:/Users/Mark/Documents/label_denoising_record_linkage/datasets/dexter" "C:/Users/Mark/Documents/label_denoising_record_linkage/datasets/evaluation_scenarios/experiment_down_stream_classifier" "C:/Users/Mark/Documents/label_denoising_record_linkage/datasets/evaluation3"

:: Loop over each target directory
for %%D in (%target_dirs%) do (
    echo Processing directory: %%~D

    :: Loop over all files in the current directory
    for %%F in (%%~D\*) do (
        set "delete=true"
        for %%W in (%whitelist%) do (
            if /I "%%~nxF"=="%%~nxW" set "delete=false"
        )
        if "!delete!"=="true" (
            echo Deleting: %%F
            del /f /q "%%F"
        )
    )
)

endlocal
