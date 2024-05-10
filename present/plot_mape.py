

import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import plotly.graph_objs as go
import plotly.express as px

# Streamlit page configuration
st.title('Crop Yield Validation MAPE Analysis')
st.write('Select the training runs to display the Validation MAPE.')
graph_names = ['mape', 'loss', 'mae']
selected_graphs = st.multiselect('Choose runs', options=graph_names, default=['mape'])


run_type = st.text_input('Enter run type', 'run_1')
# Get list of available runs
run_directories = glob.glob(f'../{run_type}/*')
run_options = [os.path.basename(run) for run in run_directories]
selected_run_dir = st.selectbox('Choose runs', options=run_options)

log_smoothing = st.checkbox('Smooth using log of MAPE')

# Function to load data from JSON log file
def load_data(run_path, value_to_find ='val_mape', train=False):
    with open(os.path.join(run_path, 'trainlog.json'), 'r') as file:
        data = json.load(file)
    epochs = list(map(int, data.keys()))
    value_to_find = 'train_'+value_to_find if train else 'val_'+value_to_find
    val_mape = [data[str(epoch)][value_to_find] for epoch in epochs]
    return epochs, val_mape


if selected_run_dir:
    crop_yield_paths = glob.glob(f'../{run_type}/{selected_run_dir}/crop_yield/*')
    run_names = [os.path.basename(run) for run in crop_yield_paths]

    # Options for datasets to filter runs
    dataset_options = ["dataset_s", "dataset_20", "dataset_50"]
    selected_dataset = st.selectbox('Choose datasets', options=dataset_options)
    
    # Allow user to choose to auto-select runs based on dataset pattern
    auto_select = st.checkbox(f'Select all runs that include "{selected_dataset}"')

    selected_runs = []
    if auto_select:
        # Automatically select runs that include the selected dataset
        selected_runs = [run for run in run_names if selected_dataset in run]
        # Update the multiselect widget with these automatically selected runs
        selected_runs = st.multiselect('Choose runs', options=run_names, default=selected_runs)
        st.write(f"Automatically selected runs: {selected_runs}")
    else:
        # Allow the user to select runs manually
        selected_runs = st.multiselect('Choose runs', options=run_names)
        st.write(f"Manually selected runs: {selected_runs}")

    if selected_runs:
        for graph_name in selected_graphs:
            fig = go.Figure()  # Initialize a plotly figure
            min_mapes = []
            val_mapes = []

            for run in selected_runs:
                full_path = os.path.join(f'../{run_type}/{selected_run_dir}/crop_yield', run)
                epochs_v, val = load_data(full_path, value_to_find=graph_name, train=False)
                epochs_t, train = load_data(full_path, value_to_find=graph_name, train=True)
                
                if log_smoothing:
                    val = [np.log(mape) if mape > 0 else 0 for mape in val]
                    train = [np.log(mape) if mape > 0 else 0 for mape in train]

                # Add traces for validation and training data
                fig.add_trace(go.Scatter(x=epochs_v[2:], y=val[2:], mode='lines+markers', name=f'Run {run} - Validation'))
                fig.add_trace(go.Scatter(x=epochs_t[2:], y=train[2:], mode='lines+markers', name=f'Run {run} - Train'))

                min_mapes.append((run, min(val)))
                val_mapes.append(val)

            # Update layout for the figure
            fig.update_layout(
                title=f'{graph_name} vs Epochs',
                xaxis_title='Epochs',
                yaxis_title='Log of ' + graph_name if log_smoothing else graph_name,
                legend_title='Legend',
                template='plotly_white'
            )

            # Display the figure in the Streamlit app
            st.plotly_chart(fig, use_container_width=True)

            # Compute the last value of MAPE for each run
            last_mapes = [(run, val_mape[-1]) for run, val_mape in zip(selected_runs, val_mapes)]

            # Combine the minimum and last MAPE values into a single list of tuples
            min_and_last_mapes = [(min_mape[0], min_mape[1], last_mape[1]) for min_mape, last_mape in zip(min_mapes, last_mapes)]

            # Display minimum and last MAPE for each run in a table
            st.write("Minimum and Last MAPE for each run:")
            st.table(min_and_last_mapes)
        

        
