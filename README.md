Streamflow Data Processing Scripts
This repository contains two Python scripts for processing and modeling streamflow data from the National Water Model (NWM) and the United States Geological Survey (USGS):

parse.py: Parses, validates, and merges raw NWM and USGS CSV files into a standardized format.

model.py: (Assumed) Script for modeling or analysis using the merged data (details TBD).

parse.py — Data Processing Pipeline
Description
parse.py processes raw CSV data from NWM and USGS sources, validates essential fields, infers missing metadata (like site numbers), and merges the datasets based on matching timestamps (DateTime). The merged data is saved as merged_data.csv.

Expected Input
Place your raw CSV files in the following folder structure:

bash
Copy
Edit
/your_project_directory/
├── parse.py
├── nwm_data/
│   ├── sample_nwm_1.csv
│   └── ...
└── usgs_data/
    ├── sample_usgs_1.csv
    └── ...
Example NWM Data (CSV)
csv
Copy
Edit
NWM_version_number,model_initialization_time,model_output_valid_time,streamflow_value,streamID
v2.1,2021-05-21_00:00:00,2021-05-21_01:00:00,3.6499999184161434,20380357
Example USGS Data (CSV)
csv
Copy
Edit
DateTime,USGSFlowValue,USGS_GageID
2021-04-20 07:00:00+00:00,32,A
Columns can also use site_no instead of USGS_GageID.

How to Run
Update the nwm_folder, usgs_folder, and output_folder paths in the __main__ block of parse.py:

python
Copy
Edit
nwm_folder = r"C:\Users\PWD\Desktop\nwm_data"
usgs_folder = r"C:\Users\PWD\Desktop\usgs_data"
output_folder = r"C:\Users\PWD\Desktop\processed_data"
Then run the script:

bash
Copy
Edit
python parse.py
Output
A file named merged_data.csv will be saved in your output_folder. It contains synchronized flow data from both NWM and USGS, merged on DateTime.

Output Format Example
csv
Copy
Edit
DateTime,streamflow,observed_flow
2021-05-21 01:00:00,3.6499999184161434,32
...
model.py — Analysis / Modeling Script
This script is expected to take the merged data (merged_data.csv) from parse.py and perform statistical analysis, modeling, or visualization (e.g., comparing modeled vs. observed flow).

Ensure merged_data.csv exists before running model.py.

Notes
Timezones in USGS data are stripped for alignment with NWM time.

Files missing required fields will be skipped with a warning.

Invalid or malformed timestamps are automatically dropped.

