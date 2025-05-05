import os
import pandas as pd

def process_file(file_path, file_type):
    """
    Reads and processes a single NWM or USGS CSV file.

    Args:
        file_path (str): The full path to the CSV file.
        file_type (str): Either 'NWM' or 'USGS'.

    Returns:
        pd.DataFrame: Processed DataFrame with standardized columns,
                      or an empty DataFrame if processing fails.
    """
    try:
        print(f"Processing {file_type} file: {file_path}")
        df = pd.read_csv(file_path)

        # Print columns to help diagnose issues
        print(f"Columns in file: {df.columns.tolist()}")

        if file_type == 'USGS':
            required_columns = ['DateTime', 'USGSFlowValue'] # Base requirements
            # Handle potential site number column names
            site_col = None
            if 'USGS_GageID' in df.columns:
                site_col = 'USGS_GageID'
            elif 'site_no' in df.columns:
                site_col = 'site_no'

            # Check for essential data columns
            missing_data_cols = [col for col in required_columns if col not in df.columns]
            if missing_data_cols:
                print(f"Warning: Missing essential data columns in {file_path}: {', '.join(missing_data_cols)}. Skipping file.")
                return pd.DataFrame() # Skip file if essential data is missing

            # If no site column found, infer from filename
            if site_col is None:
                print(f"Warning: No 'USGS_GageID' or 'site_no' column found in {file_path}. Inferring from filename.")
                try:
                    # Assumes filename format like 'SITEID_...'
                    site_no = os.path.basename(file_path).split('_')[0]
                    if not site_no.isdigit(): # Basic check if it looks like a site ID
                         raise ValueError("Extracted part is not numeric")
                    df['site_no'] = site_no
                    site_col = 'site_no'
                except Exception as e:
                    print(f"Error: Could not infer site_no from filename {file_path}: {e}. Skipping file.")
                    return pd.DataFrame()
            elif site_col == 'USGS_GageID':
                 # Rename existing column for consistency
                 df.rename(columns={'USGS_GageID': 'site_no'}, inplace=True)
                 site_col = 'site_no' # Update site_col variable


            # Select and keep only necessary columns, drop rows with NA in essential cols
            final_usgs_cols = ['DateTime', 'USGSFlowValue', site_col]
            # Add any other columns present that are needed
            available_cols = [col for col in final_usgs_cols if col in df.columns]
            df = df[available_cols]
            df = df.dropna(subset=['DateTime', 'USGSFlowValue', site_col])

        elif file_type == 'NWM':
            required_columns = ['model_output_valid_time', 'streamflow_value', 'streamID']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                # Allow flexibility if site column is named 'site_no' already
                if 'site_no' in df.columns and 'streamID' in missing_columns:
                     required_columns = ['model_output_valid_time', 'streamflow_value', 'site_no']
                     # Recheck missing after allowing 'site_no'
                     missing_columns = [col for col in required_columns if col not in df.columns]
                     if not missing_columns:
                         # If only streamID was missing but site_no exists, proceed
                         pass
                     else:
                         print(f"Warning: Missing essential columns in {file_path}: {', '.join(missing_columns)}. Skipping file.")
                         return pd.DataFrame()
                else:
                    print(f"Warning: Missing essential columns in {file_path}: {', '.join(missing_columns)}. Skipping file.")
                    return pd.DataFrame()

            # Select and keep only necessary columns, drop rows with NA
            # Handle if site column is 'site_no' instead of 'streamID'
            if 'site_no' in df.columns and 'streamID' not in df.columns:
                 df = df[['model_output_valid_time', 'streamflow_value', 'site_no']]
                 df = df.dropna(subset=['model_output_valid_time', 'streamflow_value', 'site_no'])
            else:
                 df = df[['model_output_valid_time', 'streamflow_value', 'streamID']]
                 df = df.dropna(subset=['model_output_valid_time', 'streamflow_value', 'streamID'])

        return df

    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()


def process_all_files(nwm_folder, usgs_folder, output_folder):
    # Gather all NWM and USGS CSV files from their respective folders
    nwm_files = [os.path.join(nwm_folder, file) for file in os.listdir(nwm_folder) if file.endswith('.csv')]
    usgs_files = [os.path.join(usgs_folder, file) for file in os.listdir(usgs_folder) if file.endswith('.csv')]

    # Process NWM files
    nwm_data_list = [process_file(file, 'NWM') for file in nwm_files]
    nwm_data = pd.concat([df for df in nwm_data_list if not df.empty], ignore_index=True)
    if nwm_data.empty:
        print("Error: No valid NWM data found after processing files.")
        return

    # Process USGS files
    usgs_data_list = [process_file(file, 'USGS') for file in usgs_files]
    usgs_data = pd.concat([df for df in usgs_data_list if not df.empty], ignore_index=True)
    if usgs_data.empty:
        print("Error: No valid USGS data found after processing files.")
        return

    # --- DateTime Conversion ---
    print("\n--- Preparing for Merge ---")
    nwm_data.rename(columns={'model_output_valid_time': 'DateTime', 'streamflow_value': 'streamflow'}, inplace=True)
    usgs_data.rename(columns={'USGSFlowValue': 'observed_flow'}, inplace=True)

    # Convert DateTime columns
    nwm_data['DateTime'] = pd.to_datetime(nwm_data['DateTime'].str.replace('_', ' '), errors='coerce')
    usgs_data['DateTime'] = pd.to_datetime(usgs_data['DateTime'], errors='coerce')

    # Standardize timezones: Convert both to naive datetime
    if nwm_data['DateTime'].dt.tz is not None:
        nwm_data['DateTime'] = nwm_data['DateTime'].dt.tz_localize(None)
    if usgs_data['DateTime'].dt.tz is not None:
        usgs_data['DateTime'] = usgs_data['DateTime'].dt.tz_localize(None)

    # Drop rows with invalid DateTime
    nwm_data.dropna(subset=['DateTime'], inplace=True)
    usgs_data.dropna(subset=['DateTime'], inplace=True)

    # Debugging: Print DateTime ranges
    print(f"NWM DateTime range: {nwm_data['DateTime'].min()} to {nwm_data['DateTime'].max()}")
    print(f"USGS DateTime range: {usgs_data['DateTime'].min()} to {usgs_data['DateTime'].max()}")

    # --- Merge NWM and USGS data ---
    print("\n--- Merging Data ---")
    merged_data = pd.merge(
        nwm_data[['DateTime', 'streamflow']],
        usgs_data[['DateTime', 'observed_flow']],
        on='DateTime',
        how='inner'
    )
    print(f"Merged data shape: {merged_data.shape}")

    if merged_data.empty:
        print("Warning: Merged data is empty. Check if 'DateTime' values overlap.")
        return

    # --- Save Merged Data ---
    os.makedirs(output_folder, exist_ok=True)
    merged_data.to_csv(os.path.join(output_folder, 'merged_data.csv'), index=False)
    print("Merged data saved successfully.")

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Starting Data Processing Script ---")
    # Specify your NWM and USGS folder paths (use raw strings 'r' for Windows paths)
    nwm_folder = r"C:\Users\PWD\Desktop\nwm_data"
    usgs_folder = r"C:\Users\PWD\Desktop\usgs_data"
    output_folder = r"C:\Users\PWD\Desktop\processed_data"

    print(f"NWM data folder: {nwm_folder}")
    print(f"USGS data folder: {usgs_folder}")
    print(f"Output folder: {output_folder}")

    # Run the processing pipeline
    process_all_files(nwm_folder, usgs_folder, output_folder)

    print("\n--- Data Processing Script Finished ---")