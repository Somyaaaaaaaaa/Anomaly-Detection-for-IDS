import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

def compute_reconstruction_error(autoencoder, X_test):
    reconstructions = autoencoder.predict(X_test)
    mse = np.mean(np.square(X_test- reconstructions), axis = 1)

    return reconstructions, mse

def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.savefig("outputs/plots/training_history.png")

    plt.legend()
    plt.show()

def plot_error_distribution(mse, y_test):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        mse[y_test == 0],
        label='Normal',
        fill=True
    )

    sns.kdeplot(
        mse[y_test == 1],
        label='Attack',
        fill=True
    )

    plt.xlabel("Reconstruction Error")
    plt.title("Error Distribution")
    plt.savefig("outputs/plots/error_distribution.png")
    
    plt.legend()
    plt.show()

def predict_anomalies(mse, percentile=55):
    threshold = np.percentile(mse, percentile)
    y_pred = (mse > threshold).astype(int)
    return y_pred, threshold

def evaluate_model(y_test, y_pred, mse):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, mse))
    print(classification_report(y_test, y_pred))

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("outputs/plots/confusion_matrix.png")

    plt.show()

def plot_roc_curve(y_test, mse):
    fpr, tpr, _ = roc_curve(y_test, mse)

    plt.plot(fpr, tpr, label='Autoencoder ROC Curve')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("outputs/plots/roc_curve.png")
    plt.legend()
    plt.show()

def plot_pr_curve(y_test, mse):

    auprc = average_precision_score(y_test, mse)
    print("Area Under Precision-Recall Curve (AUPRC):", auprc)

    precision, recall, thresholds = precision_recall_curve(y_test, mse)

    plt.figure(figsize=(8, 6))

    plt.plot(recall, precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig("outputs/plots/pr_curve.png")

    plt.show()

def threshold_analysis(mse, y_test):
    threshold_percentiles = range(30, 100, 5)

    precisions = []
    recalls = []
    f1_scores = []

    for p in threshold_percentiles:
        thresh = np.percentile(mse, p)
        y_predd = (mse > thresh).astype(int)
        
        precisions.append(precision_score(y_test, y_predd))
        recalls.append(recall_score(y_test, y_predd))
        f1_scores.append(f1_score(y_test, y_predd))
    
    plt.figure(figsize=(10, 6)) 
    plt.plot(threshold_percentiles, precisions, label='Precision') 
    plt.plot(threshold_percentiles, recalls, label='Recall') 
    plt.plot(threshold_percentiles, f1_scores, label='F1 Score') 
    plt.xlabel('Threshold Percentile') 
    plt.ylabel('Score') 
    plt.title('Threshold Comparison') 
    plt.legend() 
    plt.show()

def save_metrics(y_test, y_pred, mse, threshold):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, mse)
        auprc = average_precision_score(y_test,mse)
        with open("outputs/metrics/results.txt","w") as f:
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"ROC AUC Score: {roc_auc}\n")
            f.write(f"AUPRC: {auprc}\n")
        experiment_results = []
        experiment_results.append({
            "Model": "Baseline Autoencoder",
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC Score": roc_auc,
            "AUPRC": auprc
        })
        results_df = pd.DataFrame(experiment_results)
        results_df = results_df.round(3)
        print(results_df)

        results_df.to_csv("outputs/metrics/model_results.csv", index=False)