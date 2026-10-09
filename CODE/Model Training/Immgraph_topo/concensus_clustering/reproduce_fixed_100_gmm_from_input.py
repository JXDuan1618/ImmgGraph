"""Reproduce 100 fixed-seed GMM experiments from a fixed 932-feature input table."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DEFAULT_EXPERIMENT_FILE = Path(
    r"100_experiment_ids.txt"
)
DEFAULT_INPUT_FILE = Path(
    r"input_features_932d.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "fixed_100_gmm_experiments"
EXPERIMENT_PATTERN = re.compile(r"GMM_(diag|tied)_seed(\d+)")
SUMMARY_FIELDS = (
    "experiment_id",
    "configuration",
    "random_state",
    "cluster_0_n",
    "cluster_1_n",
    "cluster_2_n",
    "cluster_3_n",
    "converged",
    "n_iter",
    "lower_bound",
    "partition_hash",
    "result_directory",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition_hash(labels: np.ndarray) -> str:
    """Hash a partition after removing arbitrary numeric cluster-label identity."""
    mapping: dict[int, int] = {}
    canonical: list[int] = []
    next_label = 0
    for value in np.asarray(labels, dtype=np.int32):
        value = int(value)
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
        canonical.append(mapping[value])
    return hashlib.sha256(np.asarray(canonical, dtype=np.int8).tobytes()).hexdigest()[:16]


def load_experiments(path: Path) -> list[dict[str, int | str]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(lines) != 100 or len(set(lines)) != 100:
        raise ValueError("Experiment file must contain exactly 100 unique non-empty IDs")

    experiments = []
    for experiment_id in lines:
        match = EXPERIMENT_PATTERN.fullmatch(experiment_id)
        if match is None:
            raise ValueError(f"Invalid experiment ID: {experiment_id}")
        experiments.append(
            {
                "experiment_id": experiment_id,
                "configuration": match.group(1),
                "random_state": int(match.group(2)),
            }
        )

    configurations = (
        pd.Series([item["configuration"] for item in experiments]).value_counts().to_dict()
    )
    if configurations != {"diag": 70, "tied": 30}:
        raise ValueError(f"Expected 70 diag and 30 tied experiments, got {configurations}")
    return experiments


def load_fixed_input(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    if frame.shape != (290, 933):
        raise ValueError(f"Expected 290 rows and Patient_ID + 932 features, got {frame.shape}")
    if frame.columns[0] != "Patient_ID":
        raise ValueError(f"Expected first column to be Patient_ID, got {frame.columns[0]!r}")

    patient_ids = frame["Patient_ID"].astype(str).tolist()
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Patient_ID values are not unique")

    feature_frame = frame.drop(columns=["Patient_ID"])
    features = feature_frame.to_numpy(np.float64)
    if features.shape != (290, 932):
        raise ValueError(f"Expected 290 x 932 input, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Clustering input contains non-finite values")
    return patient_ids, features, feature_frame.columns.astype(str).tolist()


def prepare_output_dir(path: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir = path.resolve()
    results_dir = output_dir / "results"
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-file", type=Path, default=DEFAULT_EXPERIMENT_FILE)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiments = load_experiments(args.experiment_file)
    patient_ids, x_raw, feature_names = load_fixed_input(args.input_file)
    output_dir, results_dir = prepare_output_dir(args.output_dir, args.overwrite)

    scaler = StandardScaler().fit(x_raw)
    x_scaled = scaler.transform(x_raw)
    pca = PCA(n_components=0.90, svd_solver="full").fit(x_scaled)
    x_pca = pca.transform(x_scaled)

    preprocessing = {
        "scaler": scaler,
        "pca": pca,
        "feature_names": feature_names,
        "input_sources": [str(args.input_file.resolve())],
        "patient_ids": patient_ids,
    }
    joblib.dump(preprocessing, output_dir / "preprocessing.joblib", compress=3)

    pd.read_csv(args.input_file).to_csv(output_dir / "input_features_932d.csv", index=False)
    pca_input = pd.DataFrame(
        x_pca,
        columns=[f"PC{i}" for i in range(1, x_pca.shape[1] + 1)],
    )
    pca_input.insert(0, "Patient_ID", patient_ids)
    pca_input.to_csv(output_dir / "input_features_pca90.csv", index=False)
    pd.DataFrame({"Patient_ID": patient_ids}).to_csv(
        output_dir / "patient_order.csv", index=False
    )
    shutil.copy2(args.experiment_file, output_dir / "100_experiment_ids.txt")

    manifest = {
        "experiment_file": str(args.experiment_file.resolve()),
        "experiment_file_sha256": sha256_file(args.experiment_file),
        "input_file": str(args.input_file.resolve()),
        "input_file_sha256": sha256_file(args.input_file),
        "experiment_count": len(experiments),
        "configuration_counts": {"diag": 70, "tied": 30},
        "raw_input_shape": list(x_raw.shape),
        "pca_output_shape": list(x_pca.shape),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "saved_inputs": {
            "raw_features": "input_features_932d.csv",
            "pca_features": "input_features_pca90.csv",
            "preprocessing_model": "preprocessing.joblib",
            "patient_order": "patient_order.csv",
            "experiment_ids": "100_experiment_ids.txt",
        },
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "gmm_parameters": {
            "n_components": 4,
            "n_init": 1,
            "reg_covar": 1e-6,
            "init_params": "kmeans",
        },
        "reproducibility_note": (
            "This run uses the provided fixed 290 x 932 input table directly, then applies "
            "the same StandardScaler, PCA(90%, svd_solver='full'), and fixed-seed "
            "GaussianMixture settings as the supplied training script."
        ),
    }
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_rows = []
    all_assignments: dict[str, list[str] | np.ndarray] = {"Patient_ID": patient_ids}
    started = time.monotonic()
    for index, experiment in enumerate(experiments, start=1):
        experiment_id = str(experiment["experiment_id"])
        destination = results_dir / experiment_id
        destination.mkdir(exist_ok=True)

        model = GaussianMixture(
            n_components=4,
            covariance_type=str(experiment["configuration"]),
            n_init=1,
            random_state=int(experiment["random_state"]),
            reg_covar=1e-6,
            init_params="kmeans",
        )
        labels = model.fit_predict(x_pca).astype(int)
        counts = np.bincount(labels, minlength=4).tolist()
        p_hash = partition_hash(labels)

        pd.DataFrame({"Patient_ID": patient_ids, "cluster": labels}).to_csv(
            destination / "patient_assignments.csv", index=False
        )
        joblib.dump(model, destination / "gmm_model.joblib", compress=3)
        metadata = {
            **experiment,
            "cluster_sizes": counts,
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "lower_bound": float(model.lower_bound_),
            "partition_hash": p_hash,
            "preprocessing_file": "../../preprocessing.joblib",
        }
        (destination / "RESULT.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "configuration": experiment["configuration"],
                "random_state": experiment["random_state"],
                "cluster_0_n": counts[0],
                "cluster_1_n": counts[1],
                "cluster_2_n": counts[2],
                "cluster_3_n": counts[3],
                "converged": bool(model.converged_),
                "n_iter": int(model.n_iter_),
                "lower_bound": float(model.lower_bound_),
                "partition_hash": p_hash,
                "result_directory": str(destination.relative_to(output_dir)),
            }
        )
        all_assignments[experiment_id] = labels
        print(f"[{index:03d}/100] {experiment_id} sizes={counts} hash={p_hash}", flush=True)

    with (output_dir / "experiment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    pd.DataFrame(all_assignments).to_csv(
        output_dir / "all_patient_assignments.csv", index=False
    )
    print(f"Completed 100 experiments in {time.monotonic() - started:.1f} seconds", flush=True)
    print(f"Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
