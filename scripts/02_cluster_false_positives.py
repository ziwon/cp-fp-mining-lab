from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cv_fp_lab.config import load_config
from cv_fp_lab.clustering import attach_cluster_results, cluster_embeddings, reduce_umap
from cv_fp_lab.utils import ensure_dir


def main() -> None:
    cfg = load_config()
    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    metadata_file = Path(cfg["embedding"]["metadata_file"])
    embeddings_file = Path(cfg["embedding"]["output_file"])

    df = pd.read_csv(metadata_file)
    embeddings = np.load(embeddings_file)
    ccfg = cfg["clustering"]

    xy = reduce_umap(
        embeddings,
        n_neighbors=ccfg["umap_neighbors"],
        min_dist=ccfg["umap_min_dist"],
    )
    labels = cluster_embeddings(
        embeddings,
        min_cluster_size=ccfg["hdbscan_min_cluster_size"],
        min_samples=ccfg["hdbscan_min_samples"],
    )

    out_df = attach_cluster_results(df, xy, labels)
    out_df.to_csv(processed_dir / "fp_clusters.csv", index=False)
    out_df[["event_id", "umap_x", "umap_y", "cluster_id", "synthetic_fp_type"]].to_csv(
        processed_dir / "fp_umap.csv", index=False
    )
    print(out_df.groupby(["cluster_id", "synthetic_fp_type"]).size())
    print(f"Saved: {processed_dir / 'fp_clusters.csv'}")


if __name__ == "__main__":
    main()
