# InsureNet: Neural Network–Based Insurance Premium

## 🌟 Overview
A full-stack **regression pipeline** predicting **individual medical insurance charges** using a **Feedforward Neural Network** built with **PyTorch**. The workflow encompasses data ingestion, preprocessing, model training, evaluation, and real-time inference, enabling accurate premium estimation from demographic and lifestyle inputs.

---

## 🔗 Repository Information

| Detail | Value |
| :--- | :--- |
| **Repository URL** | `https://github.com/TanmoyD24/InsureNet` |

---

## 🛠️ Project Workflow

The prediction pipeline is fully contained within the `linear_Regression_model.ipynb` notebook and follows these key steps:

1.  **Data Ingestion**: Downloading the `insurance.csv` dataset from the Kaggle API.
2.  **Data Preprocessing**: Handling missing values, and transforming categorical features (`sex`, `smoker`, `region`).
3.  **Feature Engineering**: Standardizing numerical features (`age`, `bmi`, `children`) using **`StandardScaler`**.
4.  **Model Training**: Training the **InsureNet** model using PyTorch with a custom **`InsuranceDataset`** and `DataLoader`.
5.  **Evaluation & Inference**: Testing the model's predictive power using regression metrics and providing a utility for sample prediction.

---

## 🧠 Model Architecture: InsureNet

The model, defined as `SimpleNNRegressionModel`, is a custom **PyTorch Neural Network** designed for regression. 

### Architecture Details
| Layer | Description | Activation |
| :--- | :--- | :--- |
| **Input** | 6 features (Age, BMI, etc.) | - |
| **Hidden Layer 1** | Linear (6 to 128 neurons) | **ReLU** |
| **Hidden Layer 2** | Linear (128 to 256 neurons) | **ReLU** |
| **Output Layer** | Linear (256 to 1 neuron - Charge) | - |

### Training Configuration
* **Loss Function**: **`nn.L1Loss()`** (Mean Absolute Error).
* **Optimizer**: **`torch.optim.Adam`** (`lr=0.0001`).
* **Epochs**: 1000.
* **Batch Size**: 32.

---

## 🔬 Dataset & Preprocessing Summary

* **Source Dataset**: `mirichoi0218/insurance` (via Kagglehub).
* **Records**: 1,338.
* **Train/Test Split**: 80:20 ratio.

| Feature | Data Type | Preprocessing |
| :--- | :--- | :--- |
| `age`, `bmi`, `children` | Numerical | Standardized using **`StandardScaler`** |
| `sex`, `smoker`, `region` | Categorical | Encoded using **`LabelEncoder`** |
| **`charges`** (Target) | Numerical | Standardized |

---

## 📉 Model Evaluation Results

The model's performance on the test set indicates that the current simple NN architecture is insufficient for accurately modeling this dataset.

| Metric | Value | Comment |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | `161,873,504.0` | High error value. |
| **Mean Absolute Error (MAE)** | `8,370.52` | Average prediction is off by approximately $8,370. |
| **R2-Score** | **`-0.0427`** | A negative score indicates the model performs worse than simply predicting the mean charge for every input. |

---

## ▶️ Usage

To replicate the project, clone the repository and run all cells in the Jupyter Notebook:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TanmoyD24/InsureNet.git](https://github.com/TanmoyD24/InsureNet.git)
    cd InsureNet
    ```
2.  **Run the Notebook:**
    Execute the cells in `IsureNet.ipynb` to download the data, preprocess it, train the InsureNet model, and generate evaluation metrics.
