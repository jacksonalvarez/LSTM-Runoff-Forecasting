# Streamflow Data Processing Scripts

This repository contains Python scripts for processing and modeling streamflow data from the **National Water Model (NWM)** and the **United States Geological Survey (USGS)**.

## Scripts Overview

- `parse.py`: Parses, validates, and merges raw NWM and USGS CSV files into a standardized format.
- `model.py`: (Planned) Performs analysis or modeling using the merged data.

---

## `parse.py` – Data Processing Pipeline

### Description

`parse.py` processes raw CSV files from both NWM and USGS sources by:

- Validating essential columns.
- Inferring missing site identifiers from file names if necessary.
- Aligning time formats and removing invalid rows.
- Merging datasets on the `DateTime` column.
- Saving the merged result to a single CSV file.

### Expected Input Format

Place your raw CSV files into the following folder structure:

```
/your_project_directory/
├── parse.py
├── nwm_data/
│   ├── sample_nwm_1.csv
│   └── ...
└── usgs_data/
    ├── sample_usgs_1.csv
    └── ...
```

#### Example NWM CSV

```csv
NWM_version_number,model_initialization_time,model_output_valid_time,streamflow_value,streamID
v2.1,2021-05-21_00:00:00,2021-05-21_01:00:00,3.6499999184161434,20380357
```

#### Example USGS CSV

```csv
DateTime,USGSFlowValue,USGS_GageID
2021-04-20 07:00:00+00:00,32,A
```

> USGS files may use `site_no` instead of `USGS_GageID`.

---

### How to Run

1. Edit the `__main__` section in `parse.py` to point to your data folders:

```python
nwm_folder = r"C:\path\to\nwm_data"
usgs_folder = r"C:\path\to\usgs_data"
output_folder = r"C:\path\to\output_folder"
```

2. Run the script:

```bash
python parse.py
```

---

### Output

- The script creates a file called `merged_data.csv` in the specified `output_folder`.
- This file contains synchronized flow data from NWM and USGS sources.

#### Output Format Example

```csv
DateTime,streamflow,observed_flow
2021-05-21 01:00:00,3.6499999184161434,32
```

---

## `model.py` – Analysis or Modeling (Planned)

This script is intended to use `merged_data.csv` for further analysis such as:

- Statistical comparisons between modeled and observed flows.
- Visualization of time series data.
- Error metrics and performance evaluation.

> Be sure to run `parse.py` first so that `merged_data.csv` is available.

---

## Notes

- USGS timestamps may include timezone info; this is stripped during processing for consistency with NWM timestamps.
- Files missing critical columns will be skipped.
- Rows with invalid or missing `DateTime` values are dropped before merging.

---

## License

This project is open-source and free to use under the MIT License.
