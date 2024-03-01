# EEG Data Analysis README

## Overview
This Python script is designed for analyzing EEG (Electroencephalography) data obtained from experiments involving hand movements. The analysis includes preprocessing steps such as filtering, ICA (Independent Component Analysis), and epoch extraction, followed by statistical analysis using correlation coefficients and ANOVA (Analysis of Variance).

## Pre-Run Tasks
Before executing the script, ensure the following steps are completed:

1. **Upload Data Files:**
   - EEG file (.eeg)
   - VHDR file (.vhdr)
   - VMRK file (.vmrk)
   - Montage file (.bvef)

2. **Update File Paths:**
   - Set the `vhdr_file_path` variable to the path of your VHDR file.
   - Set the `montage_file_path` variable to the path of your Montage file.

3. **Define Electrodes:**
   - Specify the reference electrode, and the electrodes closest to the eyes (`eye1_elctrode` and `eye2_elctrode`).

4. **Channel Renaming (if needed):**
   - If using a custom montage, update the `renaming_channels` dictionary to match the channel names with a 32-electrode montage.

5. **Define Frequency Passband:**
   - Set `low_pass` and `high_pass` to define the frequency passband for bandpass filtering.

6. **Other Configuration Parameters:**
   - Adjust parameters such as `channels_number`, `tmin_epochs`, `tmax_epochs`, and `channel_group` based on your experimental setup.

## Running the Script
Execute the script after completing the pre-run tasks. The script performs the following steps:

1. **Data Loading:**
   - Loads the EEG data from the specified VHDR file.
   - Applies channel renaming and sets montage and reference.

2. **Filtering and Pre-Processing:**
   - Applies bandpass filtering to the data.
   - Performs ICA and plots components for manual inspection.
   - Identifies and excludes 'bad' components related to eye artifacts.
   - Generates filtered data after applying ICA reduction.

3. **Epochs Analysis:**
   - Extracts events from the recording.
   - Creates epochs objects and visualizes them.
   - Defines epochs for each block based on hand movements.

4. **Statistical Analysis:**
   - Combines epochs for the first and second halves of the EEG recording.
   - Computes correlation coefficient between the two halves.
   - Performs ANOVA for comparing individual blocks.
   - Conducts t-test for each time point and channel.

## Note
- Human judgment is required during the ICA step to determine components and channels to be removed.
- Adjust parameters and code sections related to statistical analysis based on the specific nature of your experiment.

**Important:** Refer to the experiment documentation for additional information and context related to your EEG data.
