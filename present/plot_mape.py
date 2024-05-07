import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# Streamlit page configuration
st.title('Crop Yield Validation MAPE Analysis')
st.write('Select the training runs to display the Validation MAPE.')

# Get list of available runs
seeds = glob.glob('../runs/*')
selected_seed = st.selectbox('Choose runs', options=seeds)
log_smoothing = st.checkbox('Smooth using log of MAPE')
if selected_seed:
    runs = glob.glob(selected_seed+'/crop_yield/*')
    run_names = [run.split('/')[-1] for run in runs]  # Assuming the last part is the unique name

    selected_runs = st.multiselect('Choose runs', options=run_names)

    def load_data(run_path):
        with open(f'{run_path}/trainlog.json', 'r') as file:
            data = json.load(file)
        epochs = list(map(int, data.keys()))
        val_mape = [data[str(epoch)]['val_mape'] for epoch in epochs]
        return epochs, val_mape

    # Plotting selected runs
    if selected_runs:
        plt.figure(figsize=(10, 5))
        min_mapes = []
        for run in selected_runs:
            full_path = f'{selected_seed}/crop_yield/{run}'
            epochs, val_mape = load_data(full_path)
            if log_smoothing:
                val_mape = [np.log(mape) for mape in val_mape]
            plt.plot(epochs[2:], val_mape[2:], marker='o', linestyle='-', label=f'Run {run}')
            min_mapes.append((run, min(val_mape)))
        
        plt.title('Log of Validation MAPE vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Log of Validation MAPE' if log_smoothing else 'Validation MAPE')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Display minimum MAPE for each run in a table
        st.write("Minimum MAPE for each run:")
        st.table(min_mapes)
