import streamlit as st
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# Streamlit page configuration
st.title('Crop Yield Validation MAPE Analysis')
st.write('Select the training runs to display the Validation MAPE.')

# Get list of available runs
runs = glob.glob('../runs/wacv_2024_seed0/crop_yield/*')
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
    for run in selected_runs:
        full_path = f'../runs/wacv_2024_seed0/crop_yield/{run}'
        epochs, val_mape = load_data(full_path)
        # val_mape = [np.log(mape) for mape in val_mape]
        plt.plot(epochs, val_mape, marker='o', linestyle='-', label=f'Run {run}')
        min_mape = min(val_mape)
        plt.text(epochs[-1], val_mape[-1], f'Min MAPE: {min_mape:.2f}', ha='right', va='bottom')
    
    plt.title('Log of Validation MAPE vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Log of Validation MAPE')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
