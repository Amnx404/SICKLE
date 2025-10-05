<a href="https://colab.research.google.com/drive/1vKxH3JJ6TLv63y3kwTZ7VQzVo2EJPZqg#scrollTo=1mbkt9ohRPDh" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# Enhancing Adaptive Multi-Sensor Satellite Fusion for Robust Agricultural Prediction

This repository contains the official implementation for our research on adaptive multi-sensor fusion for agricultural prediction. Our work began with the creation of the **SICKLE dataset (WACV 2024)** to provide a robust foundation for multi-sensor analysis. We then developed an advanced **Multi-Head Cross Fusion model (WACV 2025 Submission)** that significantly outperforms baseline models, especially under challenging conditions like cloud cover.

# Our final Approach

<img width="1170" height="777" alt="image" src="https://github.com/user-attachments/assets/fed84c22-3159-4e53-8bf0-c656adb36f57" />





**[SICKLE Website](https://sites.google.com/iiitd.ac.in/sickle/) | [SICKLE Paper (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/html/Sani_SICKLE_A_Multi-Sensor_Satellite_Imagery_Dataset_Annotated_With_Multiple_Key_WACV_2024_paper.html) | [SICKLE Video](https://www.youtube.com/watch?v=2p4BDVLrmdw) | [Request Dataset Access](https://docs.google.com/forms/d/e/1FAIpQLSdq7Dcj5FF1VmlKozrQ7XNoq006iVKrUIMTK2jReBJDuO1N2g/viewform)**

---

## The Challenge & Our Solution 🎯

Accurate agricultural forecasting using satellite imagery is often hindered by real-world challenges. A primary issue is **cloud cover**, which obscures the ground and makes optical data from satellites like Sentinel-2 and Landsat-8 unreliable. While initial models on our SICKLE dataset used simple concatenation, this method struggles when one data source is compromised.

To solve this, we developed a novel **Multi-Head Cross Fusion** architecture. This model moves beyond simple data merging and instead learns to **adaptively fuse** information from multiple sensors, dynamically prioritizing the most reliable data at any given time.



### Architecture Deep Dive 🛠️

Our model's strength comes from its sophisticated fusion pipeline that intelligently combines radar and optical data.

1.  **Spatio-Temporal Encoding**: First, time-series data from each satellite (**Sentinel-1**, **Sentinel-2**, **Landsat-8**) is processed by its own encoder to extract deep spatial and temporal features.

2.  **Cross-Attention Mechanism**: This is the core of our model. Instead of just combining features, we use cross-attention to make them interact. The features from one satellite (e.g., Sentinel-1) are used to query and refine the features from the other satellites (e.g., Sentinel-2 and Landsat-8). This enriches each satellite's data with context from the others, creating a much more powerful representation.

3.  **Adaptive Fusion**: The enhanced feature maps are then combined using a multi-head attention module. This allows the model to weigh the importance of each data source. If Sentinel-2 data is cloudy (less informative), the model automatically learns to rely more heavily on the clean radar data from Sentinel-1 to make its prediction.

---

## Performance & Results 📊

We validated our model on the SICKLE dataset against a standard concatenation baseline. The results confirm our model is more accurate and significantly more resilient to data loss.

### Performance on Full Dataset

When trained on the complete dataset, our Cross Fusion model set a new state-of-the-art for crop type, harvest date, and crop yield prediction.

| Task         | Metric | Concatenation Model | **Our Cross Fusion Model** |
| :----------- | :----- | :------------------ | :------------------------- |
| **Crop Type** | IoU%   | $81.07 \pm 5.77$    | **$86.25 \pm 2.31$** |
| **Harvest Date** | MAE    | $10.75 \pm 3.39$    | **$8.63 \pm 0.35$** |
| **Crop Yield** | MAPE%  | $49.63 \pm 7.95$    | **$46.06 \pm 0.72$** |

### Robustness Under Cloud Cover

We simulated cloud cover by removing **50% of the optical data** (Sentinel-2, Landsat-8) from the training set. Our fusion model demonstrated remarkable resilience, maintaining high accuracy where the baseline model struggled.

| Task         | Metric    | Concatenation Model | **Our Cross Fusion Model** |
| :----------- | :-------- | :------------------ | :------------------------- |
| **Crop Type** | IoU%      | 87.85               | **90.59** |
| **Harvest Date**| MAE       | 10.06               | **8.89** |

Notably, our fusion model's performance on the **occluded dataset** was still superior to the baseline model trained on the **full dataset** for harvest date prediction (8.89 MAE vs. 10.75 MAE). This proves its effectiveness in real-world, imperfect conditions.

---

## Getting Started

### Dataset Download
Please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSdq7Dcj5FF1VmlKozrQ7XNoq006iVKrUIMTK2jReBJDuO1N2g/viewform) to gain access to the SICKLE dataset, pre-trained weights, and other files.

### Dependencies
Install all dependencies using the following command:
```bash
pip install -r requirements.txt
````

### Inference with Pre-trained Models

After receiving access, download and unzip `weights.zip` into the root directory. You will have a `runs` folder. Use the following commands to run inference.

  * **Example (UTAE Fusion)**: `./evaluate.sh <path_to_data> [S1,S2,L8] utae`

### Training Models from Scratch

Use the following commands to train a model from scratch.

  * **Example (UTAE Fusion)**: `./train.sh <path_to_data> [S1,S2,L8] utae`

-----

## Citation

If you use our dataset or models in your research, please cite our papers:

```bibtex
@InProceedings{Sani_2024_WACV,
    author    = {Sani, Depanshu and Mahato, Sandeep and Saini, Sourabh and Agarwal, Harsh Kumar and Devshali, Charu Chandra and Anand, Saket and Arora, Gaurav and Jayaraman, Thiagarajan},
    title     = {SICKLE: A Multi-Sensor Satellite Imagery Dataset Annotated With Multiple Key Cropping Parameters},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5995-6004}
}

@inproceedings{Anonymous_2025_WACV,
    author    = {Anonymous Authors},
    title     = {Enhancing Adaptive Multi-Sensor Satellite Fusion for Robust Agricultural Prediction},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2025}
}
```

## Acknowledgments

  - This work was partly supported by Google’s AI for Social Good “Impact Scholars” program and the Infosys Center for Artificial Intelligence at IIIT-Delhi.
  - We thank Parichya Sirohi for contributions in the early stages of the project.
  - We express our gratitude to Dr. Gopinath R. and Dr. Rajakumar R. from Ecotechnology, MS Swaminathan Research Foundation, for their valuable inputs and assistance with field data collection.
  - This work builds upon the implementation provided by [Garnot et al.](https://github.com/VSainteuf/utae-paps).

<!-- end list -->
