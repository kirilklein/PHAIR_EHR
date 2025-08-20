import os
import json
import warnings
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import umap
from corebehrt.constants.data import PID_COL

# Suppress common warnings from visualization libraries
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


class EncodingAnalyzer:
    """
    A standalone class to load, analyze, and visualize patient encodings.

    It takes a directory of saved encodings, performs a comprehensive analysis including
    statistical summaries, similarity metrics, and dimensionality reduction,
    and saves the results as plots and a structured JSON file.
    """

    def __init__(self, encoding_dir: str, save_dir: str, pid_col: str = PID_COL):
        """
        Initializes the EncodingAnalyzer.

        Args:
            encoding_dir (str): The directory containing 'patient_encodings.parquet'.
            save_dir (str): The directory where analysis results (plots, JSON) will be saved.
            pid_col (str): The name of the patient identifier column in the parquet file.
        """
        self.encoding_dir = encoding_dir
        self.save_dir = save_dir
        self.pid_col = pid_col

        # Define file paths
        self.patient_encodings_path = os.path.join(
            self.encoding_dir, "patient_encodings.parquet"
        )
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize data and result holders
        self.patient_encodings_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.pids: Optional[np.ndarray] = None
        self.analysis_results = {}
        self.embeddings_pca: Optional[np.ndarray] = None
        self.embeddings_umap: Optional[np.ndarray] = None
        self.embeddings_tsne: Optional[np.ndarray] = None
        self.pca_model = None

        print(f"EncodingAnalyzer initialized. Reading from: {self.encoding_dir}")
        print(f"Analysis results will be saved to: {self.save_dir}")

    def _load_data(self):
        """Loads patient encodings from the specified parquet file."""
        print("\n--- Loading Data ---")
        if not os.path.exists(self.patient_encodings_path):
            raise FileNotFoundError(
                f"Could not find patient encodings file at: {self.patient_encodings_path}"
            )

        self.patient_encodings_df = pd.read_parquet(self.patient_encodings_path)
        print(f"Loaded patient encodings with shape: {self.patient_encodings_df.shape}")

        feature_cols = [
            col for col in self.patient_encodings_df.columns if col.startswith("x")
        ]
        self.embeddings = self.patient_encodings_df[feature_cols].values
        self.pids = self.patient_encodings_df[self.pid_col].values
        print(f"Extracted embeddings matrix with shape: {self.embeddings.shape}")

    def _compute_statistics(self, sample_size: int = 1000):
        """Computes basic, similarity, and per-dimension statistics."""
        print("\n--- Computing Statistics ---")
        self.analysis_results["embedding_stats"] = {
            "shape": self.embeddings.shape,
            "mean": float(self.embeddings.mean()),
            "std": float(self.embeddings.std()),
            "min": float(self.embeddings.min()),
            "max": float(self.embeddings.max()),
        }

        # Sample for efficient pairwise calculations
        if len(self.embeddings) > sample_size:
            sample_indices = np.random.choice(
                len(self.embeddings), sample_size, replace=False
            )
            sample_embeddings = self.embeddings[sample_indices]
        else:
            sample_embeddings = self.embeddings

        euclidean_dists = euclidean_distances(sample_embeddings)
        cosine_sims = cosine_similarity(sample_embeddings)
        mask = ~np.eye(euclidean_dists.shape[0], dtype=bool)  # Exclude self-similarity

        self.euclidean_dists_flat = euclidean_dists[mask]
        self.cosine_sims_flat = cosine_sims[mask]

        self.analysis_results["similarity_stats"] = {
            "mean_euclidean_distance": float(self.euclidean_dists_flat.mean()),
            "std_euclidean_distance": float(self.euclidean_dists_flat.std()),
            "mean_cosine_similarity": float(self.cosine_sims_flat.mean()),
            "std_cosine_similarity": float(self.cosine_sims_flat.std()),
        }

        dim_stds = self.embeddings.std(axis=0)
        self.analysis_results["dimension_stats"] = {
            "dimension_means": self.embeddings.mean(axis=0).tolist(),
            "dimension_stds": dim_stds.tolist(),
            "highest_variance_dims": np.argsort(dim_stds)[-5:].tolist(),
            "lowest_variance_dims": np.argsort(dim_stds)[:5].tolist(),
        }

    def _perform_dimensionality_reduction(self, n_pca_components: int = 50):
        """Performs PCA, UMAP, and t-SNE."""
        print("\n--- Performing Dimensionality Reduction ---")
        # PCA is run first for efficiency and as an analysis tool
        print("Computing PCA...")
        self.pca_model = PCA(n_components=n_pca_components)
        embeddings_pca_full = self.pca_model.fit_transform(self.embeddings)
        self.embeddings_pca = embeddings_pca_full[:, :2]  # For scatter plot

        self.analysis_results["pca_stats"] = {
            "explained_variance_ratio_top10": self.pca_model.explained_variance_ratio_[
                :10
            ].tolist(),
            "cumulative_variance_top10": np.cumsum(
                self.pca_model.explained_variance_ratio_[:10]
            ).tolist(),
        }

        # UMAP and t-SNE are run on the PCA-reduced data for speed
        print("Computing UMAP...")
        umap_reducer = umap.UMAP(
            n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
        )
        self.embeddings_umap = np.asarray(
            umap_reducer.fit_transform(embeddings_pca_full)
        )

        print("Computing t-SNE...")
        tsne_reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        self.embeddings_tsne = np.asarray(
            tsne_reducer.fit_transform(embeddings_pca_full)
        )

    def _generate_main_visualization(self):
        """Generates and saves the first 2x3 summary plot."""
        print("Generating main visualization...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Patient Encoding Analysis Summary", fontsize=16)

        # Dimensionality Reduction Plots
        scatter1 = axes[0, 0].scatter(
            self.embeddings_umap[:, 0],
            self.embeddings_umap[:, 1],
            c=range(len(self.embeddings_umap)),
            cmap="viridis",
            alpha=0.7,
            s=20,
        )
        axes[0, 0].set_title("UMAP Projection")
        axes[0, 0].set_xlabel("UMAP 1")
        axes[0, 0].set_ylabel("UMAP 2")
        fig.colorbar(scatter1, ax=axes[0, 0], label="Sample Index")

        scatter2 = axes[0, 1].scatter(
            self.embeddings_tsne[:, 0],
            self.embeddings_tsne[:, 1],
            c=range(len(self.embeddings_tsne)),
            cmap="plasma",
            alpha=0.7,
            s=20,
        )
        axes[0, 1].set_title("t-SNE Projection")
        axes[0, 1].set_xlabel("t-SNE 1")
        axes[0, 1].set_ylabel("t-SNE 2")
        fig.colorbar(scatter2, ax=axes[0, 1], label="Sample Index")

        scatter3 = axes[0, 2].scatter(
            self.embeddings_pca[:, 0],
            self.embeddings_pca[:, 1],
            c=range(len(self.embeddings_pca)),
            cmap="coolwarm",
            alpha=0.7,
            s=20,
        )
        axes[0, 2].set_title("PCA Projection (First 2 Components)")
        axes[0, 2].set_xlabel("PC 1")
        axes[0, 2].set_ylabel("PC 2")
        fig.colorbar(scatter3, ax=axes[0, 2], label="Sample Index")

        # Distribution Plots
        mean_euc = self.euclidean_dists_flat.mean()
        axes[1, 0].hist(
            self.euclidean_dists_flat, bins=50, color="lightcoral", edgecolor="black"
        )
        axes[1, 0].set_title("Distribution of Pairwise Euclidean Distances")
        axes[1, 0].axvline(
            mean_euc, color="red", linestyle="--", label=f"Mean: {mean_euc:.3f}"
        )
        axes[1, 0].legend()

        mean_cos = self.cosine_sims_flat.mean()
        axes[1, 1].hist(
            self.cosine_sims_flat, bins=50, color="goldenrod", edgecolor="black"
        )
        axes[1, 1].set_title("Distribution of Pairwise Cosine Similarities")
        axes[1, 1].axvline(
            mean_cos, color="red", linestyle="--", label=f"Mean: {mean_cos:.3f}"
        )
        axes[1, 1].legend()

        axes[1, 2].plot(
            range(1, 21), self.pca_model.explained_variance_ratio_[:20], "bo-"
        )
        axes[1, 2].set_title("PCA Explained Variance Ratio")
        axes[1, 2].set_xlabel("Principal Component")
        axes[1, 2].set_ylabel("Explained Variance Ratio")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.save_dir, "patient_encodings_analysis.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved main analysis plot to {save_path}")

    def _generate_detailed_visualization(self, n_dims_to_show=20):
        """Generates and saves the second 2x2 detailed plot."""
        print("Generating detailed visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Detailed Embedding Structure Analysis", fontsize=16)

        # Heatmap of a sample of patient embeddings
        n_patients_sample = min(50, len(self.embeddings))
        sns.heatmap(
            self.embeddings[:n_patients_sample, :n_dims_to_show],
            ax=axes[0, 0],
            cmap="RdBu_r",
            center=0,
        )
        axes[0, 0].set_title(f"Embeddings Heatmap (First {n_patients_sample} Patients)")
        axes[0, 0].set_xlabel("Embedding Dimension")
        axes[0, 0].set_ylabel("Patient Index")

        # Distribution of all embedding values
        mean_val = self.embeddings.mean()
        axes[0, 1].hist(
            self.embeddings.flatten(), bins=100, color="indianred", edgecolor="black"
        )
        axes[0, 1].set_title("Distribution of All Embedding Values")
        axes[0, 1].set_xlabel("Embedding Value")
        axes[0, 1].axvline(
            mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}"
        )
        axes[0, 1].legend()

        # Box plot of distributions per dimension
        axes[1, 0].boxplot(self.embeddings[:, :n_dims_to_show], patch_artist=True)
        axes[1, 0].set_title(f"Distribution Across First {n_dims_to_show} Dimensions")
        axes[1, 0].set_xlabel("Embedding Dimension")
        axes[1, 0].set_ylabel("Value")

        # Correlation matrix between dimensions
        corr_matrix = np.corrcoef(self.embeddings[:, :n_dims_to_show].T)
        sns.heatmap(corr_matrix, ax=axes[1, 1], cmap="RdBu_r", center=0, square=True)
        axes[1, 1].set_title(f"Correlation Matrix (First {n_dims_to_show} Dims)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(
            self.save_dir, "patient_encodings_detailed_analysis.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved detailed analysis plot to {save_path}")

    def _save_numerical_results(self):
        """Saves the computed statistics to a JSON file."""
        save_path = os.path.join(self.save_dir, "encoding_analysis_results.json")
        with open(save_path, "w") as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"Saved numerical results to {save_path}")

    def _print_summary(self):
        """Prints a final summary and collapse assessment to the console."""
        print("\n" + "=" * 20 + " ANALYSIS SUMMARY " + "=" * 20)
        print(
            f"Mean pairwise cosine similarity: {self.analysis_results['similarity_stats']['mean_cosine_similarity']:.4f}"
        )
        print(
            f"Variance explained by first PC: {self.analysis_results['pca_stats']['explained_variance_ratio_top10'][0]:.4f}"
        )

        print("\n=== COLLAPSE ASSESSMENT ===")
        if self.analysis_results["similarity_stats"]["mean_cosine_similarity"] > 0.9:
            print(
                "⚠️  HIGH SIMILARITY: Embeddings may be experiencing representation collapse (anisotropy)."
            )
        elif self.analysis_results["similarity_stats"]["mean_cosine_similarity"] < 0.1:
            print(
                "✅ EXCELLENT SEPARATION: Embeddings show very low average similarity."
            )
        else:
            print(
                "✅ GOOD SEPARATION: Embeddings appear to have reasonable separation."
            )

        if (
            self.analysis_results["pca_stats"]["explained_variance_ratio_top10"][0]
            > 0.25
        ):
            print(
                "⚠️  LOW INTRINSIC DIMENSIONALITY: A few components explain most of the variance."
            )
        else:
            print(
                "✅ HIGH INTRINSIC DIMENSIONALITY: Variance is well-distributed across components."
            )
        print("=" * 60)

    def analyze(self):
        """
        The main public method to run the entire analysis pipeline.
        """
        try:
            self._load_data()
            self._compute_statistics()
            self._perform_dimensionality_reduction()
            self._generate_main_visualization()
            self._generate_detailed_visualization()
            self._save_numerical_results()
            self._print_summary()
        except Exception as e:
            print(f"\nAn error occurred during analysis: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    """
    Example of how to run the analyzer from the command line.
    
    Usage:
    python encoding_analyzer.py /path/to/your/encodings /path/to/your/output
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and visualize patient encodings."
    )
    parser.add_argument(
        "encoding_dir",
        type=str,
        help="Directory containing the 'patient_encodings.parquet' file.",
    )
    parser.add_argument(
        "save_dir", type=str, help="Directory where analysis results will be saved."
    )
    parser.add_argument(
        "--pid_col",
        type=str,
        default="subject_id",
        help="Name of the patient ID column.",
    )

    args = parser.parse_args()

    analyzer = EncodingAnalyzer(
        encoding_dir=args.encoding_dir, save_dir=args.save_dir, pid_col=args.pid_col
    )
    analyzer.analyze()
