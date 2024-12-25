import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import CustomDataset


class PostModelingReport:
    def __init__(self, model, tokenizer, tokenizer_config, eval_dataset, experiment_location, device):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.eval_dataset = eval_dataset
        self.experiment_location = experiment_location
        self.device = device
        self.plot_dir = os.path.join(experiment_location, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

    def _extract_embeddings(self, dataset):
        with torch.no_grad():
            embeddings = []
            for example in tqdm(dataset, desc="Extracting embeddings"):
                pred = self.model.base_model(example['input_ids'].unsqueeze(0).to(self.device)).last_hidden_state[0, 0, :].detach().cpu().numpy()
                embeddings.append(pred)
            return np.array(embeddings)

    def plot_loss(self, trainer):
        log_history = trainer.state.log_history
        training_losses = [entry['loss'] for entry in log_history if 'loss' in entry]
        eval_losses = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]

        plt.plot(training_losses, label="Training Loss")
        plt.plot(eval_losses, label="Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Over Epochs")
        plt.savefig(f"{self.plot_dir}/loss_plot.png", format="png", dpi=300)

    def plot_embeddings_similarity(self):
        long_names = pd.read_csv('/home/david/Desktop/projects/NameEmbedding/data/testing/close_names_long.csv')
        long_names_shorter = long_names.iloc[:25, :]
        long_names_dataset = CustomDataset(
            text_list=long_names_shorter['name'].tolist(),
            tokenizer=self.tokenizer,
            max_len=self.tokenizer_config['max_len'],
            include_attention_mask=True
        )

        long_names_predictions = self._extract_embeddings(long_names_dataset)

        # UMAP and Clustering
        reducer = umap.UMAP(n_neighbors=50)
        embedding = reducer.fit_transform(self._extract_embeddings(self.eval_dataset))
        long_names_embeddings = reducer.transform(long_names_predictions)

        long_names_embeddings_pd = pd.DataFrame(long_names_embeddings)
        long_names_embeddings_pd['group'] = long_names['group']

        embedding_pd = pd.DataFrame(embedding)
        embedding_pd['group'] = 'all_the_rest'
        embedding_pd = pd.concat([embedding_pd, long_names_embeddings_pd], axis=0, ignore_index=True)

        # Clustering
        dbscan = DBSCAN(eps=1.0, min_samples=5)
        labels = dbscan.fit_predict(embedding_pd[[0, 1]])
        embedding_pd['cluster'] = labels

        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=embedding_pd, x=0, y=1, hue='group', alpha=0.5)
        plt.title("UMAP - Embeddings with Clustering")
        plt.savefig(f"{self.plot_dir}/clustering.png", format="png", dpi=300)

        # Individual Cluster Plots
        cluster_plots = self.create_cluster_plots(embedding_pd)
        for cluster_id, cluster_data in cluster_plots.items():
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=cluster_data, x=0, y=1, hue='group', alpha=0.8)
            plt.title(f"Cluster {cluster_id} - Size: {len(cluster_data)}", fontsize=14)
            plt.savefig(f"{self.plot_dir}/cluster_{cluster_id}.png", format="png", dpi=300)
            plt.close()

    def create_cluster_plots(self, embedding_pd, num_cluster_plots=4):
        # Perform clustering and get largest clusters
        dbscan = DBSCAN(eps=1.0, min_samples=5)
        labels = dbscan.fit_predict(embedding_pd[[0, 1]])
        embedding_pd['cluster'] = labels
        
        # Filter valid clusters (non-noise, i.e., label != -1) and sort by size
        cluster_sizes = embedding_pd[embedding_pd['cluster'] != -1]['cluster'].value_counts()
        largest_clusters = cluster_sizes.index[:num_cluster_plots]  # Get the largest clusters
        
        cluster_plots = {}
        for cluster_id in largest_clusters:
            cluster_plots[cluster_id] = embedding_pd[embedding_pd['cluster'] == cluster_id]
        
        return cluster_plots

    def k_fold_classification(self, dataset, labels):
        embeddings = self._extract_embeddings(dataset)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        labels = torch.tensor(labels)
        for train_index, val_index in kf.split(embeddings):
            train_embeddings, val_embeddings = embeddings[train_index], embeddings[val_index]
            train_labels, val_labels = labels[train_index], labels[val_index]

            # Train classifier
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(train_embeddings, train_labels)

            # Evaluate
            val_preds = classifier.predict(val_embeddings)
            accuracy = accuracy_score(val_labels, val_preds)
            accuracies.append(accuracy)

        avg_accuracy = np.mean(accuracies)
        return avg_accuracy

    def plot_group_cluster_distribution(self, group_cluster_analysis):
        # Prepare the data for plotting
        cluster_data = pd.DataFrame({
            'Group': list(group_cluster_analysis.keys()),
            'Number of Clusters': list(group_cluster_analysis.values())
        })

        # Sort the data by the number of clusters in increasing order
        cluster_data = cluster_data.sort_values(by='Number of Clusters', ascending=True)

        # Create the bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=cluster_data, x='Group', y='Number of Clusters', palette='viridis')

        plt.xlabel("Group", fontsize=14)
        plt.ylabel("Number of Clusters", fontsize=14)
        plt.title("Number of Clusters per Group of Names", fontsize=16)
        plt.xticks(rotation=90, ha='center')  # Rotate group names by 90 degrees for better readability
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{self.plot_dir}/group_cluster_distribution.png", format="png", dpi=300)
        plt.close()

    def generate_post_modeling_report(self, trainer):
        print('Starting Post modeling report')

        # Plot Loss
        self.plot_loss(trainer)

        # Plot Embeddings Similarity
        self.plot_embeddings_similarity()

        # Load Word/Object dataset for classification
        df = pd.read_csv('./data/testing/name_vs_object.csv')
        words = df['Word'].values
        labels = df['Type'].apply(lambda x: 1 if x == 'name' else 0).values

        # Run K-Fold classification
        avg_accuracy = self.k_fold_classification(
            CustomDataset(
                text_list=words.tolist(),
                tokenizer=self.tokenizer,
                max_len=self.tokenizer_config['max_len'],
                include_attention_mask=True
            ),
            labels
        )
        print(f"Avg accuracy for word/object classifier: {avg_accuracy:.4f}")

        # Save KPIs
        KPI_table = {
            'KPI_name': ['Name_Object_ACC'],
            'KPI_value': [avg_accuracy]
        }

        # Calculate NMI and Average Clusters per Group
        long_names = pd.read_csv('/home/david/Desktop/projects/NameEmbedding/data/testing/close_names_long.csv')
        # Use the full dataset instead of slicing to include all names
        long_names_dataset = CustomDataset(
            text_list=long_names['name'].tolist(),
            tokenizer=self.tokenizer,
            max_len=self.tokenizer_config['max_len'],
            include_attention_mask=True
        )
        long_names_predictions = self._extract_embeddings(long_names_dataset)

        reducer = umap.UMAP(n_neighbors=50)
        long_names_embeddings = reducer.fit_transform(long_names_predictions)

        # DBSCAN for NMI and Average Clusters per Group
        long_names_embeddings_pd = pd.DataFrame(long_names_embeddings)
        long_names_embeddings_pd['group'] = long_names['group']

        dbscan = DBSCAN(eps=1.0, min_samples=5)
        labels = dbscan.fit_predict(long_names_embeddings_pd[[0, 1]])
        long_names_embeddings_pd['cluster'] = labels

        group_cluster_distribution = {}
        for group_name, group_data in long_names_embeddings_pd.groupby('group'):
            valid_clusters = group_data[group_data['cluster'] != -1]['cluster'].unique()
            group_cluster_distribution[group_name] = list(valid_clusters)

        # Analyze group-cluster distribution
        group_cluster_analysis = {
            group: len(clusters) for group, clusters in group_cluster_distribution.items()
        }

        # Plot group cluster distribution
        self.plot_group_cluster_distribution(group_cluster_analysis)

        nmi_score = normalized_mutual_info_score(long_names_embeddings_pd['group'], long_names_embeddings_pd['cluster'])
        average_clusters_per_group = np.mean(list(group_cluster_analysis.values()))

        # Add to KPI table
        KPI_table['KPI_name'].extend(['NMI_close_names', 'average_clusters_per_group'])
        KPI_table['KPI_value'].extend([nmi_score, average_clusters_per_group])

        # Save KPIs to CSV
        pd.DataFrame(KPI_table).to_csv(f"{self.plot_dir}/KPIs.csv", index=False)

        print("Post modeling report generated successfully.")

