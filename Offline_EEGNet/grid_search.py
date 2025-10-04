#!/usr/bin/env python3
"""
Grid search over hyperparameters for EEGNet training.
Searches: window_length, num_noise, num_folds, stride
"""

import os
import sys
import itertools
import pandas as pd
import gc
import torch
from shared_utils import utils
from pipeline_kf_func import pipeline
import shutil

# Define hyperparameter grid
PARAM_GRID = {
    "window_length": [125, 150, 200, 250],
    "num_noise": [4, 6, 8],
    "num_folds": [3, 5, 7],
    "stride": [12, 18],
}

TOP_K_MODELS = 5  # Only keep top 5 models


def run_grid_search(base_config_path, output_dir="grid_search_results"):
    """
    Run grid search over hyperparameters.

    Args:
        base_config_path: Path to base YAML config file
        output_dir: Directory to store results
    """
    # Load base config
    base_config = utils.read_config(base_config_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate all combinations
    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[name] for name in param_names]
    combinations = list(itertools.product(*param_values))

    print(f"Starting grid search with {len(combinations)} combinations...")
    print(f"Parameters: {param_names}")
    print("=" * 80)

    results = []

    for idx, params in enumerate(combinations):
        # Create param dict
        param_dict = dict(zip(param_names, params))

        print(f"\n[{idx + 1}/{len(combinations)}] Testing parameters:")
        for name, value in param_dict.items():
            print(f"  {name}: {value}")

        # Modify config
        config = base_config.copy()
        config["dataset_generator"]["window_length"] = param_dict["window_length"]
        config["augmentation"]["window_length"] = param_dict["window_length"]
        config["augmentation"]["num_noise"] = param_dict["num_noise"]
        config["augmentation"]["stride"] = param_dict["stride"]
        config["partition"]["num_folds"] = param_dict["num_folds"]
        config["training"]["num_folds"] = param_dict["num_folds"]
        config["model"]["window_length"] = param_dict["window_length"]

        # Generate unique model name
        model_name = f"grid_{idx}_w{param_dict['window_length']}_n{param_dict['num_noise']}_f{param_dict['num_folds']}_s{param_dict['stride']}"

        # Compute model_dir before try block
        model_dir = (
            config["model_dir"]
            + model_name
            + "_"
            + "_".join(config["data_names"])
            + "/"
        )

        try:
            # Run pipeline
            pipeline(config, model_name=model_name, train_kf=True)

            # Extract results from CSV
            results_csv = model_dir + "results.csv"

            if os.path.exists(results_csv):
                df = pd.read_csv(results_csv)
                mean_val_acc = df["Validation Acc"].mean()
                mean_train_acc = df["Training Acc"].mean()
                max_val_acc = df["Validation Acc"].max()

                print(f"  Mean Val Acc: {mean_val_acc:.4f}")
                print(f"  Max Val Acc: {max_val_acc:.4f}")
                print(f"  Mean Train Acc: {mean_train_acc:.4f}")
            else:
                print(f"  ERROR: Results file not found!")
                mean_val_acc = 0.0
                mean_train_acc = 0.0
                max_val_acc = 0.0

            # Store results
            result_entry = param_dict.copy()
            result_entry["mean_val_acc"] = mean_val_acc
            result_entry["max_val_acc"] = max_val_acc
            result_entry["mean_train_acc"] = mean_train_acc
            result_entry["model_name"] = model_name
            result_entry["model_dir"] = model_dir
            results.append(result_entry)

        except Exception as e:
            print(f"  ERROR: Training failed with exception: {e}")
            result_entry = param_dict.copy()
            result_entry["mean_val_acc"] = 0.0
            result_entry["max_val_acc"] = 0.0
            result_entry["mean_train_acc"] = 0.0
            result_entry["model_name"] = model_name
            result_entry["model_dir"] = model_dir
            result_entry["error"] = str(e)
            results.append(result_entry)

        # Rolling cleanup: Keep only top K models as we go
        if len(results) > TOP_K_MODELS:
            results_sorted = sorted(
                results, key=lambda x: x["mean_val_acc"], reverse=True
            )
            kth_best_acc = results_sorted[TOP_K_MODELS - 1]["mean_val_acc"]
            current_acc = results[-1]["mean_val_acc"]

            if current_acc < kth_best_acc:
                print(
                    f"  → Deleting (below top {TOP_K_MODELS}: {current_acc:.4f} < {kth_best_acc:.4f})"
                )
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
            else:
                models_to_keep = set(
                    [r["model_dir"] for r in results_sorted[:TOP_K_MODELS]]
                )
                for result in results:
                    result_dir = result.get("model_dir", "")
                    if (
                        result_dir
                        and os.path.exists(result_dir)
                        and result_dir not in models_to_keep
                    ):
                        print(f"  → Deleting: {os.path.basename(result_dir)}")
                        shutil.rmtree(result_dir)

        # Clean up memory after each run to prevent memory leak
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "grid_search_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)

    # Sort by mean validation accuracy
    results_df_sorted = results_df.sort_values("mean_val_acc", ascending=False)

    # No cleanup needed - already done during training
    print(f"\nKept top {TOP_K_MODELS} models (cleaned up during training)")

    print(f"\nResults saved to: {results_csv_path}")
    print("\nTop 10 configurations by mean validation accuracy:")
    print(results_df_sorted.head(10).to_string(index=False))

    # Print best configuration
    best_row = results_df_sorted.iloc[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    print(f"  window_length: {int(best_row['window_length'])}")
    print(f"  num_noise: {int(best_row['num_noise'])}")
    print(f"  num_folds: {int(best_row['num_folds'])}")
    print(f"  stride: {int(best_row['stride'])}")
    print(f"  Mean Validation Accuracy: {best_row['mean_val_acc']:.4f}")
    print(f"  Max Validation Accuracy: {best_row['max_val_acc']:.4f}")
    print(f"  Mean Training Accuracy: {best_row['mean_train_acc']:.4f}")
    print(f"  Model Name: {best_row['model_name']}")
    print("=" * 80)

    return results_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python grid_search.py [config.yaml]")
        print("Default: config_gpu.yaml")
        base_config_path = "config_gpu.yaml"
    else:
        base_config_path = sys.argv[1]

    run_grid_search(base_config_path)
