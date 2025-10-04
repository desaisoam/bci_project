#!/usr/bin/env python3
"""
Sequential hyperparameter search - much faster than grid search!
Optimizes one parameter at a time: window_length → num_noise → num_folds → stride
Total: ~12 runs instead of 72
"""

import os
import sys
import pandas as pd
import gc
import torch
from shared_utils import utils
from pipeline_kf_func import pipeline
import shutil

# Define search ranges for each parameter
PARAM_RANGES = {
    "window_length": [125, 150, 200, 250],
    "num_noise": [2, 4, 8],
    "num_folds": [3, 5, 7],
    "stride": [12, 18],
}

# Starting defaults (will be updated as we find better values)
BEST_PARAMS = {"window_length": 125, "num_noise": 4, "num_folds": 5, "stride": 12}


def cleanup_memory():
    """Aggressively clean up memory between runs"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_single_experiment(config, param_name, param_value, best_params, idx):
    """Run a single training experiment"""
    # Update config with current best params + the one we're testing
    test_params = best_params.copy()
    test_params[param_name] = param_value

    config["dataset_generator"]["window_length"] = test_params["window_length"]
    config["augmentation"]["window_length"] = test_params["window_length"]
    config["augmentation"]["num_noise"] = test_params["num_noise"]
    config["augmentation"]["stride"] = test_params["stride"]
    config["partition"]["num_folds"] = test_params["num_folds"]
    config["training"]["num_folds"] = test_params["num_folds"]
    config["model"]["window_length"] = test_params["window_length"]

    # Generate model name
    model_name = f"seq_{idx}_{param_name}{param_value}_w{test_params['window_length']}_n{test_params['num_noise']}_f{test_params['num_folds']}_s{test_params['stride']}"
    model_dir = (
        config["model_dir"] + model_name + "_" + "_".join(config["data_names"]) + "/"
    )

    print(f"\n{'=' * 80}")
    print(f"Testing {param_name} = {param_value}")
    print(f"Current params: {test_params}")
    print(f"Model: {model_name}")
    print(f"{'=' * 80}\n")

    try:
        # Run pipeline
        pipeline(config, model_name=model_name, train_kf=True)

        # Extract results
        results_csv = model_dir + "results.csv"
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            mean_val_acc = df["Validation Acc"].mean()
            print(f"✓ Mean Validation Accuracy: {mean_val_acc:.4f}")
        else:
            print(f"✗ Results file not found!")
            mean_val_acc = 0.0

        # Cleanup memory immediately after run
        cleanup_memory()

        return {
            "param_name": param_name,
            "param_value": param_value,
            "mean_val_acc": mean_val_acc,
            "model_name": model_name,
            "model_dir": model_dir,
            **test_params,
        }

    except Exception as e:
        print(f"✗ Training failed: {e}")
        cleanup_memory()
        return {
            "param_name": param_name,
            "param_value": param_value,
            "mean_val_acc": 0.0,
            "model_name": model_name,
            "model_dir": model_dir,
            "error": str(e),
            **test_params,
        }


def sequential_search(base_config_path, output_dir="sequential_search_results"):
    """
    Run sequential hyperparameter search.
    Optimizes parameters one at a time in order: window_length → num_noise → num_folds → stride
    """
    # Load base config
    base_config = utils.read_config(base_config_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    best_params = BEST_PARAMS.copy()
    run_idx = 0

    print("\n" + "=" * 80)
    print("SEQUENTIAL HYPERPARAMETER SEARCH")
    print("=" * 80)
    print(f"Starting with defaults: {best_params}")
    print(f"Search order: window_length → num_noise → num_folds → stride")

    # Optimize each parameter sequentially
    for param_name in ["window_length", "num_noise", "num_folds", "stride"]:
        print(f"\n\n{'#' * 80}")
        print(f"# OPTIMIZING: {param_name}")
        print(f"# Current best params: {best_params}")
        print(f"{'#' * 80}\n")

        param_results = []

        # Test each value for this parameter
        for param_value in PARAM_RANGES[param_name]:
            result = run_single_experiment(
                base_config.copy(), param_name, param_value, best_params, run_idx
            )
            param_results.append(result)
            all_results.append(result)
            run_idx += 1

        # Find best value for this parameter
        best_result = max(param_results, key=lambda x: x["mean_val_acc"])
        best_params[param_name] = best_result["param_value"]

        print(f"\n{'=' * 80}")
        print(f"BEST {param_name}: {best_result['param_value']}")
        print(f"Validation Accuracy: {best_result['mean_val_acc']:.4f}")
        print(f"Updated best params: {best_params}")
        print(f"{'=' * 80}\n")

        # Delete non-best models immediately
        print(f"Cleaning up non-best models for {param_name}...")
        best_model_dir = best_result["model_dir"]
        for result in param_results:
            model_dir = result.get("model_dir", "")
            if model_dir and model_dir != best_model_dir and os.path.exists(model_dir):
                print(f"  → Deleting: {os.path.basename(model_dir)}")
                shutil.rmtree(model_dir)

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_dir, "sequential_search_results.csv")
    results_df.to_csv(results_csv_path, index=True)

    # Clean up - keep only models from final optimized run
    print(f"\nCleaning up intermediate models...")
    final_model_dirs = set()

    # Keep the best model from each parameter optimization step
    for param_name in ["window_length", "num_noise", "num_folds", "stride"]:
        param_results = [r for r in all_results if r["param_name"] == param_name]
        best_result = max(param_results, key=lambda x: x["mean_val_acc"])
        final_model_dirs.add(best_result["model_dir"])

    # Delete all other models
    for result in all_results:
        model_dir = result.get("model_dir", "")
        if os.path.exists(model_dir) and model_dir not in final_model_dirs:
            print(f"  Deleting: {model_dir}")
            shutil.rmtree(model_dir)

    print(
        f"\nCleanup complete. Kept {len(final_model_dirs)} best models (one per parameter)."
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("SEQUENTIAL SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {results_csv_path}")
    print(f"\nFINAL OPTIMIZED PARAMETERS:")
    print(f"  window_length: {best_params['window_length']}")
    print(f"  num_noise: {best_params['num_noise']}")
    print(f"  num_folds: {best_params['num_folds']}")
    print(f"  stride: {best_params['stride']}")

    # Show progression
    print(f"\nOPTIMIZATION PROGRESSION:")
    for param_name in ["window_length", "num_noise", "num_folds", "stride"]:
        param_results = [r for r in all_results if r["param_name"] == param_name]
        best_result = max(param_results, key=lambda x: x["mean_val_acc"])
        print(
            f"  {param_name}: {best_result['param_value']} (Val Acc: {best_result['mean_val_acc']:.4f})"
        )

    print("=" * 80)

    return results_df, best_params


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sequential_search.py [config.yaml]")
        print("Default: config_gpu.yaml")
        base_config_path = "config_gpu.yaml"
    else:
        base_config_path = sys.argv[1]

    results_df, best_params = sequential_search(base_config_path)
