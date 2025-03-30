# Multi-Class ECG Heartbeat Classification

[Adam David](https://www.adamdavid.dev)

---

## Project Overview  
This project aims to classify ECG heartbeats as either a normal heartbeat or myocardial infarction using deep learning techniques. The model utilizes convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to process time-series data effectively.

The dataset used is the **ECG200 Time Series** dataset from the Time Series Machine Learning Website. This dataset was formatted and donated by R. Olszewski. Each series traces the electrical activity recorded during one heartbeat. 

---

## Dataset  
**Link:** [ECG200 Heartbeat](https://dl.acm.org/doi/book/10.5555/935627)  

### Features  
1. Time-Series Data:
- Raw ECG signal with a length of 96

### Target  
The first column of the CSV file is the target variable, indicating whether it is a normal heartbeat or myocardial infarction. 
- **1:** Normal Heartbeat  
- **-1:** Myocardial Infarction  

## Project Structure  
This project contains the following key components:  
- **Preprocessing:**
    - Data Normalization and Augmentation
    - Batching of time-series data using a custom data generator
    - One-hot encoding for label compatibility with softmax activation
- **Best Convolutional Neural Network Architecture Found:**  
    - **Input Layer**: 96 neurons (representing the time series of length 96)  
    - **Convolution/Hidden Layers**:
        - Convolution 1: filters: 1->8, k=6, s=1, p=0, ReLU activation
        - Convolution 2: filters: 8->32, k=3, s=2, p=0, ReLU activation
        - Max Pooling Layer: poolSize=2, s=2, p=0
        - Dense 1: 300 neurons, ReLU activation  
        - Dense 2: 100 neurons, ReLU activation   
    - **Output Layer**:  
        - 2 neurons, Softmax activation (for multi-class classification)  
    - **Optimizer**: Adaptive Moment Estimation (Adam)  
    - **Loss Function**: Categorical Cross-Entropy Loss 
    - **Metrics**: Accuracy and Loss
- **Best Recurrent Neural Network Architecure Found:**
    - **Input Layer**: 96 neurons (representing the time series of length 96)  
    - **Recurrent/Hidden Layers**:
        - Long Short-Term Memory Layer: 16 hidde states
        - Dense 1: 32 neurons, TanH activation  
        - Dense 2: 16 neurons, TanH activation   
    - **Output Layer**:  
        - 2 neurons, Softmax activation (for multi-class classification)  
    - **Optimizer**: Root Mean Square Propogation (RMSProp)
    - **Loss Function**: Categorical Cross-Entropy Loss 
    - **Metrics**: Accuracy and Loss
- **Model Training and Evaluation:**  
    - Uses 50-25-25 train-validation-test split
    - Batch Size of 16 is optimal for both CNN and RNN
    - Performance metrics include Accuracy and Loss

## Installation and Requirements  

```bash
# Clone the repository
git clone https://github.com/adxmd/ecg_heartbeat_classification.git

# Navigate to the project directory
cd ecg_heartbeat_classification

# Install the required dependencies
pip install -r requirements.txt
pip install tensorflow scikit-learn pandas matplotlib
```

## How to Run

1. Place the dataset training, validation and testing CSV files in your project directory
2. Modify the file path in the script accordingly
3. Run the script using: 

```bash
python3 ecg_heartbeat_classification.py
```


## Results

When computing the performance of the best CNN and RNN models mentioned above on the test dataset, we see the CNN achieve a consistent performance of **~90% accuracy** and the RNN achieves a consistent accuracy of **~84%**.

This project includes a function called `parameterTuning()` that systematically tests different hyperparameter combinations to evaluate the model's performance.

When experimenting with Convolutional Neural Networks, I tested how different kernel sizes and epochs trained affect performance.
-`Kernel Sizes: [2, 4, 6, 8]`
-`Epochs: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]`

Similar to CNNs, when experimenting with Recurrent Neural Networks, I tested many different combinations of types of recurrent layers, the number of hidden units for recurrent cells, and the number of epochs trained, and this data gave me insights into how the model is performing.
-`Types of RNN: [LSTM, GRU]`
-`RNN Hidden States: [16, 32, 48, 64, 72, 96]`
-`Epochs: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]`

The learning rate is fixed at 0.01. However, further experiments could be done on how modifying the learning rate will affect performance. 

Visualizations are generated using `Matplotlib` to compare how the accuracy and loss are affected by different hyperparameters and how many epochs the model is trained for. 
<!-- ![alt text](/thyroid-cancer-recurrence/results_moreLR.png) -->
<!--
![alt text](https://github.com/adxmd/thyroid-cancer-recurrence/blob/main/results_moreLR.png?raw=true)

Based on this visualization we can conclude that for this current neural network architecture, `Learning Rates: 0.01, 0.05, and 0.1` provide the best accuracy and loss values. They can predict recurrence with **~95% accuracy** after training for 40 epochs
--> 

