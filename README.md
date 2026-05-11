# Network Intrusion Detection using Deep Autoencoders

A fully unsupervised deep learning-based intrusion detection framework that uses reconstruction-based anomaly detection and latent-space analysis to identify anomalous network behavior.

---

## Overview

Traditional signature-based intrusion detection systems struggle against evolving and previously unseen attack patterns. This project explores a fully unsupervised anomaly detection approach where a deep autoencoder is trained exclusively on normal network traffic and learns compressed behavioral representations of legitimate activity.

When malicious traffic deviates from learned normal behavior, reconstruction error increases significantly, allowing anomalous network events to be detected without relying on predefined attack signatures.

In addition to anomaly detection, this project investigates the internal latent-space structure learned by the encoder using clustering analysis, PCA-based visualization, and silhouette scoring.

The system was implemented and evaluated using the NSL-KDD intrusion detection dataset.

---

## Key Features

* Fully unsupervised anomaly detection using deep autoencoders
* Reconstruction-error-based intrusion detection
* Latent-space representation learning
* PCA visualization of compressed traffic representations
* KMeans clustering analysis in latent space
* Silhouette score evaluation for cluster quality
* Threshold sensitivity analysis
* ROC and Precision-Recall analysis
* Modular ML pipeline architecture
* Automated experiment artifact generation

---

## System Architecture

```text
NSL-KDD Dataset
        ↓
Data Preprocessing
        ↓
Feature Encoding & Normalization
        ↓
Normal Traffic Extraction
        ↓
Deep Autoencoder Training
        ↓
Latent Representation Learning
        ↓
Reconstruction Error Computation
        ↓
Statistical Thresholding
        ↓
Evaluation + Latent Space Analysis
```

---

## Dataset

Dataset used:

* NSL-KDD Intrusion Detection Dataset

Files:

* `KDDTrain+.txt`
* `KDDTest+.txt`

The dataset contains:

* normal network traffic
* denial-of-service (DoS) attacks
* probing attacks
* privilege escalation attacks
* remote access attacks

---

## Model Architecture

### Encoder

Input → 64 → 32 → 16

### Bottleneck Representation

8-dimensional latent space

### Decoder

16 → 32 → 64 → Output

Dropout regularization was applied during encoding to improve generalization and reduce overfitting.

---

## Training Strategy

* Loss Function: Mean Squared Error (MSE)
* Optimizer: Adam
* Early Stopping Enabled
* Validation Split: 20%
* Training Data: Normal traffic only

The model learns compressed representations of legitimate traffic behavior and identifies deviations during inference.

---

## Detection Methodology

1. Train the autoencoder exclusively on normal traffic.
2. Reconstruct unseen network traffic samples.
3. Compute reconstruction error using Mean Squared Error.
4. Apply statistical thresholding based on reconstruction-error distribution.
5. Flag high-error samples as anomalies.

Threshold sensitivity analysis was additionally performed using percentile-based threshold variations to study precision-recall tradeoffs under different anomaly sensitivity conditions.

---

## Evaluation Metrics

The following metrics were used to evaluate detection performance:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* AUPRC

---

## Results

### Detection Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.827 |
| Precision | 0.940 |
| Recall    | 0.743 |
| F1 Score  | 0.830 |
| ROC-AUC   | 0.942 |
| AUPRC     | 0.963 |

The model demonstrated strong anomaly detection capability despite operating in a fully unsupervised setting.

Additional threshold sensitivity and clustering analysis further demonstrated the effectiveness of the learned latent representations for behavioral characterization of network traffic.

---

## Latent Space Analysis

The encoder’s bottleneck representations were further analyzed to investigate whether meaningful behavioral structures emerged in latent space.

Techniques used:

* PCA dimensionality reduction
* KMeans clustering
* Silhouette analysis

This analysis suggests that the encoder learns partially separable internal representations of network behaviors and attack characteristics.

---

## Visualizations

Generated outputs include:

* Training and validation loss curves
* Reconstruction error distributions
* Confusion matrix
* ROC curve
* Precision-Recall curve
* Threshold sensitivity analysis
* PCA latent-space visualization
* KMeans cluster visualization
* Silhouette score analysis

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Somyaaaaaaaaa/Latent-Space-IDS.git
cd autoencoder
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the full pipeline:

```bash
python main.py
```

The system will:

* preprocess data
* train the autoencoder
* evaluate anomaly detection performance
* generate visualizations
* perform latent-space analysis
* save outputs automatically

---

## Future Improvements

Potential extensions include:

* LSTM-based sequential intrusion detection
* Real-time packet stream analysis
* Adaptive thresholding strategies
* Variational autoencoders
* Ensemble anomaly detection
* Attention-based architectures
* Online learning for evolving attack behavior
* Automated continuous traffic monitoring pipelines
* Explainable AI for latent-space interpretability

---

## Research Motivation

Modern cyber threats evolve faster than static signature databases can adapt. This project explores how unsupervised representation learning can contribute toward more adaptive intrusion detection systems capable of identifying previously unseen attack behaviors.

The project additionally investigates latent-space organization and behavioral representation learning in order to better understand how deep autoencoder architectures internally characterize anomalous network traffic patterns.

---

## Author

**Somya**
BSc Data Science
Deep Learning • Cybersecurity • Representation Learning
