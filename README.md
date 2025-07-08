
## Network Anomaly Detection using Autoencoders and Reinforcement Learning

This project explores an advanced approach to detecting anomalies in network traffic by combining the power of Autoencoders and Reinforcement Learning (RL). It aims to build a hybrid system where an autoencoder is used to reduce the dimensionality of input data and highlight anomalies, while reinforcement learning agents refine the detection process based on feedback.

### 📌 Project Goals

* *Dimensionality Reduction*: Leverage autoencoders to compress high-dimensional network traffic features.
* *Anomaly Detection*: Identify patterns that deviate from normal network behavior.
* *Reinforcement Learning Integration*: Train agents to make detection decisions based on environment rewards.

---

### 📁 Project Structure


Network_Anomaly_Detection.ipynb
README.md
models/
    └── saved_autoencoder.h5
data/
    └── network_traffic.csv


---

### 🚀 Technologies Used

* Python
* Jupyter Notebook
* TensorFlow / Keras (for autoencoders)
* OpenAI Gym / Custom RL environment
* NumPy, Pandas, Scikit-learn (for preprocessing)
* Matplotlib, Seaborn (for visualization)

---

### 📊 Dataset

The dataset used for this project contains labeled network traffic data with a mix of normal and anomalous connections. Features include:

* Duration
* Protocol type
* Source/Destination bytes
* Flag types
* ...and others.

---

### 🧠 Models

#### 1. *Autoencoder*

* Input Layer → Encoding Layer(s) → Bottleneck → Decoding Layer(s)
* Loss Function: Mean Squared Error (MSE)
* Output: Reconstruction error used to determine anomaly score.

#### 2. *Reinforcement Learning Agent*

* Environment: Custom reward system based on false positives/negatives.
* Algorithm: Deep Q-Learning or Proximal Policy Optimization (PPO)
* Goal: Learn policies that improve detection confidence.

---

### 🧪 How to Run

1. Clone the repository:

   bash
   git clone https://github.com/kumarpiyushraj/Network-Anomaly-Detection-using-RL-model-and-Autoencoders.git
   cd Network_Anomaly_Detection
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   

3. Run the notebook:

   bash
   jupyter notebook Network_Anomaly_Detection.ipynb
   

---

### 📈 Results

* ROC-AUC Score: \~0.94
* Precision-Recall balanced for high-risk anomaly detection
* RL agents reduced false positives over training epochs

---

### 🔍 Future Work

* Apply variational autoencoders for probabilistic outputs.
* Try other RL models like A3C or DDPG.
* Deploy real-time anomaly detection in network simulations.
