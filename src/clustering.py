
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os

def run_clustering(features_file='data/features.csv', output_dir='results'):
    print("=== Unsupervised Clustering Analysis ===")
    
    if not os.path.exists(features_file):
        print(f"Error: {features_file} not found.")
        return
        
    # Load Data
    df = pd.read_csv(features_file)
    X = df.drop(['label', 'filename'], axis=1).values
    y = df['label'].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode Labels for coloring
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    # 1. K-Means
    print("Running K-Means (k=10)...")
    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)
    
    
    # 2. t-SNE
    print("Running t-SNE dimensionality reduction...")
    # n_iter is default 1000, explicitly removing it to avoid error in newer sklearn
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    plot_df['Genre'] = y
    plot_df['Cluster'] = clusters
    
    # Plot 1: True Genres
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=plot_df, x='TSNE1', y='TSNE2', hue='Genre', 
        palette='tab10', style='Genre', alpha=0.7, s=60
    )
    plt.title('t-SNE Visualization of Music Genres (True Labels)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/tsne_genres.png")
    plt.close()
    
    # Plot 2: K-Means Clusters
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=plot_df, x='TSNE1', y='TSNE2', hue='Cluster', 
        palette='viridis', alpha=0.7, s=60, legend='full'
    )
    plt.title('t-SNE Visualization of K-Means Clusters (k=10)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_clusters.png")
    plt.close()
    
    print(f"Clustering plots saved to {output_dir}/tsne_genres.png and {output_dir}/tsne_clusters.png")

    # Plot 3: KDE Density Plot (Overlapping Blobs)
    print("Generating KDE density plot...")
    plt.figure(figsize=(12, 10))
    # Filter for top genres if too crowded, or plot all. plotting all for now.
    # We use kdeplot with fill=True to create "blobs"
    sns.kdeplot(
        data=plot_df, x='TSNE1', y='TSNE2', hue='Genre', 
        fill=True, alpha=0.3, palette='tab10', levels=5, thresh=0.2
    )
    plt.title('KDE Density Visualization of Music Genres (Blobs)')
    # Move legend out
    # Note: kdeplot usually handles legend automatically with hue, but access might need adjustment if complex
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kde_genres.png")
    plt.close()
    
    print(f"KDE plot saved to {output_dir}/kde_genres.png")

if __name__ == "__main__":
    run_clustering()
