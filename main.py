from src.preprocessing import load_data,preprocess_data,scale_data,extract_normal_data
from src.model import build_autoencoder
from src.train import train_autoencoder
from src.evaluation import compute_reconstruction_error,plot_training_history, plot_error_distribution,predict_anomalies,evaluate_model,plot_confusion_matrix,plot_roc_curve,plot_pr_curve,threshold_analysis, save_metrics
from src.latent_analysis import extract_latent_vectors,reduce_dimensions,perform_clustering,plot_true_labels,plot_clusters,compute_silhouette_scores,plot_silhouette_scores
train_df, test_df = load_data(
    "data/raw/KDDTrain+.txt",
    "data/raw/KDDTest+.txt"
)

train_df, test_df = preprocess_data(train_df,test_df)
print(train_df["label"].value_counts())

X_train, X_test, y_test, train_df, test_df = scale_data(train_df,test_df)
X_training_set = extract_normal_data(X_train, train_df)

input_dim = X_training_set.shape[1]
autoencoder, encoder = build_autoencoder(input_dim)

history = train_autoencoder(autoencoder, X_training_set)
plot_training_history(history)
reconstructions, mse = compute_reconstruction_error(autoencoder, X_test)

plot_error_distribution(mse,y_test)

y_pred, threshold = predict_anomalies(mse,percentile=55)

evaluate_model(y_test, y_pred, mse)

plot_confusion_matrix(y_test,y_pred)

plot_roc_curve(y_test, mse)

plot_pr_curve(y_test,mse)

threshold_analysis(mse,y_test)

latent_vectors = extract_latent_vectors(encoder,X_test)

latent_2d = reduce_dimensions(latent_vectors)

clusters = perform_clustering(latent_vectors,n_clusters=8)

plot_true_labels(latent_2d, test_df)

plot_clusters(latent_2d, clusters)

cluster_range, scores = compute_silhouette_scores(latent_vectors)

plot_silhouette_scores(cluster_range,scores)

save_metrics(y_test, y_pred, mse, threshold)

autoencoder.save(
    "outputs/models/autoencoder.keras"
)