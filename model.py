import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import hydroeval as he
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# --- Configuration ---
input_file = 'processed_data/merged_data.csv'
lead_times = range(1, 19)  # Lead times in hours
sequence_length = 6
hidden_size = 50
num_layers = 2
learning_rate = 0.001
num_epochs = 50
batch_size = 32
train_val_split_date = datetime(2022, 9, 30)
test_start_date = datetime(2022, 10, 1)
test_end_date = datetime(2023, 4, 30)

# --- Define the LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# --- Load and Preprocess Data ---
print("Loading and preprocessing data...")
df = pd.read_csv(input_file, parse_dates=['DateTime'])
df = df.sort_values(by=['DateTime'])
df = df.drop_duplicates(subset=['DateTime'], keep='first')
df = df.dropna()

# --- Prepare data for each lead time ---
all_lead_time_data = {}
for lead_time in lead_times:
    lead_time_df = df[['DateTime', 'streamflow', 'observed_flow']].copy()
    lead_time_df['target_datetime'] = lead_time_df['DateTime'] + pd.Timedelta(hours=lead_time)
    lead_time_df = pd.merge(
        lead_time_df[['DateTime', 'streamflow', 'target_datetime']],
        df[['DateTime', 'observed_flow']],
        left_on='target_datetime', right_on='DateTime', how='inner',
        suffixes=('_now', '_future')
    )
    lead_time_df.rename(
        columns={
            'DateTime_now': 'DateTime',
            'streamflow': 'nwm_forecast',
            'observed_flow_future': 'observed_flow',
        },
        inplace=True,
    )
    lead_time_df = lead_time_df.drop(columns=['DateTime_future', 'target_datetime'])
    lead_time_df = lead_time_df.sort_values(by=['DateTime'])
    all_lead_time_data[lead_time] = lead_time_df

# --- Train, Validate, and Test for each lead time ---
models = {}
scalers = {}
all_corrected_forecasts = {lt: [] for lt in lead_times}
all_nwm_test = {lt: [] for lt in lead_times}
all_observed_test = {lt: [] for lt in lead_times}
test_datetimes_all = {lt: [] for lt in lead_times}

for lead_time, lt_df in all_lead_time_data.items():
    print(f"\n--- Processing Lead Time: {lead_time} hours ---")

    nwm_scaler = MinMaxScaler()
    observed_scaler = MinMaxScaler()

    nwm_scaled = nwm_scaler.fit_transform(lt_df[['nwm_forecast']])
    observed_scaled = observed_scaler.fit_transform(lt_df[['observed_flow']])

    features = np.array([nwm_scaled[i:i + sequence_length] for i in range(len(nwm_scaled) - sequence_length)])
    targets = np.array([observed_scaled[i + sequence_length] for i in range(len(observed_scaled) - sequence_length)])
    datetimes = lt_df['DateTime'].values[sequence_length:]
    nwm_original = lt_df['nwm_forecast'].values[sequence_length:]
    observed_original = lt_df['observed_flow'].values[sequence_length:]

    train_val_mask = datetimes <= np.datetime64(train_val_split_date)
    test_mask = (datetimes >= np.datetime64(test_start_date)) & (datetimes <= np.datetime64(test_end_date))

    train_val_features = torch.tensor(features[train_val_mask], dtype=torch.float32)
    train_val_targets = torch.tensor(targets[train_val_mask], dtype=torch.float32)
    test_features = torch.tensor(features[test_mask], dtype=torch.float32)
    test_targets = torch.tensor(targets[test_mask], dtype=torch.float32)
    test_datetimes = datetimes[test_mask]
    nwm_test_original = nwm_original[test_mask]
    observed_test_original = observed_original[test_mask]

    # Split train_val
    train_features, val_features, train_targets, val_targets = train_test_split(
        train_val_features, train_val_targets, test_size=0.2, shuffle=False
    )

    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and train the model
    input_size = 1
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training model...")
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader):.4f}")

    # Testing
    model.eval()
    predicted_scaled = []
    with torch.no_grad():
        for test_inputs, _ in test_loader:
            outputs = model(test_inputs)
            predicted_scaled.extend(outputs.cpu().numpy())

    predicted_flow = observed_scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).flatten()

    models[lead_time] = model
    scalers[f'nwm_{lead_time}'] = nwm_scaler
    scalers[f'observed_{lead_time}'] = observed_scaler
    all_corrected_forecasts[lead_time].extend(predicted_flow)
    all_nwm_test[lead_time].extend(nwm_test_original)
    all_observed_test[lead_time].extend(observed_test_original)
    test_datetimes_all[lead_time].extend(test_datetimes)


# --- Evaluate and Plot ---
print("\n--- Evaluating and Plotting ---")
all_observed_for_plot = []
all_nwm_for_plot = []
all_corrected_for_plot = []
nwm_metrics = {'CC': [], 'RMSE': [], 'PBIAS': [], 'NSE': []}
corrected_metrics = {'CC': [], 'RMSE': [], 'PBIAS': [], 'NSE': []}
lead_time_labels = []
rmse_data = []

with PdfPages('boxplot_outputs.pdf') as pdf:
    # 1. Box plots for each hour interval (1-18)
    for lead_time in lead_times:
        observed = np.array(all_observed_test[lead_time])
        nwm_forecast = np.array(all_nwm_test[lead_time])
        corrected_forecast = np.array(all_corrected_forecasts[lead_time])

        plt.figure(figsize=(10, 6))
        box_data_lt = [observed, nwm_forecast, corrected_forecast]
        labels_lt = ['Observed', 'NWM', 'Corrected']
        sns.boxplot(data=box_data_lt)
        plt.xticks(range(len(labels_lt)), labels_lt)
        plt.ylabel("Runoff Value")
        plt.title(f"Runoff Comparison - Lead Time: {lead_time} hours")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        all_observed_for_plot.append(observed)
        all_nwm_for_plot.append(nwm_forecast)
        all_corrected_for_plot.append(corrected_forecast)
        lead_time_labels.append(f"{lead_time}h")

        # Calculate metrics
        if len(observed) > 1:
            cc_nwm = np.corrcoef(observed, nwm_forecast)[0, 1] if np.std(observed) != 0 and np.std(nwm_forecast) != 0 else np.nan
            rmse_nwm = np.sqrt(mean_squared_error(observed, nwm_forecast))
            pbias_nwm = he.pbias(nwm_forecast, observed)
            nse_nwm = he.nse(nwm_forecast, observed)

            cc_corrected = np.corrcoef(observed, corrected_forecast)[0, 1] if np.std(observed) != 0 and np.std(corrected_forecast) != 0 else np.nan
            rmse_corrected = np.sqrt(mean_squared_error(observed, corrected_forecast))
            pbias_corrected = he.pbias(corrected_forecast, observed)
            nse_corrected = he.nse(corrected_forecast, observed)

            nwm_metrics['CC'].append(cc_nwm)
            nwm_metrics['RMSE'].append(rmse_nwm)
            nwm_metrics['PBIAS'].append(pbias_nwm)
            nwm_metrics['NSE'].append(nse_nwm)

            corrected_metrics['CC'].append(cc_corrected)
            corrected_metrics['RMSE'].append(rmse_corrected)
            corrected_metrics['PBIAS'].append(pbias_corrected)
            corrected_metrics['NSE'].append(nse_corrected)

            rmse_data.append({
                'Lead Time (hours)': lead_time,
                'RMSE NWM': rmse_nwm,
                'RMSE Corrected': rmse_corrected,
            })
        else:
            for metric in nwm_metrics:
                nwm_metrics[metric].append(np.nan)
            for metric in corrected_metrics:
                corrected_metrics[metric].append(np.nan)
            rmse_data.append({
                'Lead Time (hours)': lead_time,
                'RMSE NWM': np.nan,
                'RMSE Corrected': np.nan,
            })

    # 2. Box plots for overall data comparison
    plt.figure(figsize=(10, 6))
    overall_observed = np.concatenate(all_observed_for_plot)
    overall_nwm = np.concatenate(all_nwm_for_plot)
    overall_corrected = np.concatenate(all_corrected_for_plot)
    overall_box_data = [overall_observed, overall_nwm, overall_corrected]
    overall_labels = ['Observed', 'NWM', 'Corrected']  # ,'USGS Site 1', 'USGS Site 2'] # Add more labels if you have more USGS data
    sns.boxplot(data=overall_box_data)
    plt.xticks(range(len(overall_labels)), overall_labels)
    plt.ylabel("Runoff Value")
    plt.title("Overall Runoff Comparison")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # (3) Box-plots of Evaluation Metrics
    metrics = ['CC', 'RMSE', 'PBIAS', 'NSE']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_values = [nwm_metrics[metric], corrected_metrics[metric]]
        labels = ['NWM', 'Corrected']
        sns.boxplot(data=metric_values, ax=ax)
        ax.set_xticklabels(labels)
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.set_title(f'Box-plot of {metric}')
        ax.set_xticks([0, 1])
    fig.suptitle('Evaluation Metrics Comparison Across Lead Times', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

# --- Save RMSE Data to CSV ---
rmse_df = pd.DataFrame(rmse_data)
rmse_df.to_csv('rmse_comparison.csv', index=False)

print("\n--- Model Training, Evaluation, and Plotting Completed ---")
print("Box plot outputs saved to boxplot_outputs.pdf")
print("RMSE comparison for each lead time saved to rmse_comparison.csv")