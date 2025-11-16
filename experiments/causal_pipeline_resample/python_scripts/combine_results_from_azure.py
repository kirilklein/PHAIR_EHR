#!/usr/bin/env python3
"""
Combine estimate results from Azure ML mounted dataset.

This script mounts an Azure ML dataset and uses combine_results.py to merge
all estimate_results.csv files into a single combined CSV.

Usage:
    python combine_results_from_azure.py \
        --datastore researcher_data \
        --dataset_path AKK/experiments/trace/simulation/two_effects/bert/v01 \
        --output combined_results.csv \
        --print_summary
"""

import argparse
import sys
from pathlib import Path

try:
    from azureml.core import Workspace, Dataset, Datastore
except ImportError:
    print("Error: azureml-core is not installed.")
    print("Install it with: pip install azureml-core")
    sys.exit(1)

# Import the combination function from combine_results.py
from combine_results import combine_estimate_results


def mount_and_combine(
    datastore_name: str,
    dataset_path: str,
    output_path: str,
    workspace_config: str = None,
    estimators: list = None,
    print_summary: bool = False,
    upload_to_azure: bool = False,
):
    """
    Mount Azure ML dataset and combine estimate results.

    Args:
        datastore_name: Name of the Azure ML datastore
        dataset_path: Path within the datastore to the results directory
        output_path: Local path to save the combined CSV
        workspace_config: Optional path to config.json file
        estimators: Optional list of estimators to filter ['baseline', 'bert']
        print_summary: Whether to print summary statistics
        upload_to_azure: Whether to upload results back to Azure storage
    """

    print("=" * 80)
    print("AZURE ML DATASET MOUNTING AND COMBINATION")
    print("=" * 80)

    # 1. Connect to Azure ML Workspace
    print("\n[1/4] Connecting to Azure ML Workspace...")
    try:
        if workspace_config:
            workspace = Workspace.from_config(path=workspace_config)
        else:
            workspace = Workspace.from_config()
        print(f"  ✓ Connected to workspace: {workspace.name}")
    except Exception as e:
        print(f"  ✗ Error connecting to workspace: {e}")
        print("\nMake sure you have a config.json file or specify --workspace_config")
        sys.exit(1)

    # 2. Get Datastore and Create Dataset
    print(f"\n[2/4] Accessing datastore: {datastore_name}...")
    try:
        datastore = Datastore.get(workspace, datastore_name)
        print(f"  ✓ Got datastore: {datastore_name}")
    except Exception as e:
        print(f"  ✗ Error accessing datastore: {e}")
        sys.exit(1)

    print(f"\n[3/4] Creating dataset from path: {dataset_path}...")
    try:
        dataset = Dataset.File.from_files(path=(datastore, dataset_path))
        print(f"  ✓ Dataset created")
    except Exception as e:
        print(f"  ✗ Error creating dataset: {e}")
        sys.exit(1)

    # 3. Mount and Process
    print(f"\n[4/4] Mounting dataset and combining results...")
    try:
        with dataset.mount() as mount_context:
            mounted_path = mount_context.mount_point
            print(f"  ✓ Mounted at: {mounted_path}")

            # Run the combination
            print(f"\nRunning combination on mounted path...")
            combined_df = combine_estimate_results(
                results_dir=mounted_path, fit_methods=estimators
            )

            if combined_df.empty:
                print("  ✗ No results found in mounted dataset")
                sys.exit(1)

            # Print summary if requested
            if print_summary:
                print("\n" + "=" * 80)
                print("SUMMARY")
                print("=" * 80)
                print(f"\nTotal rows: {len(combined_df)}")

                if "simulation_run" in combined_df.columns:
                    print(
                        f"Simulation runs: {sorted(combined_df['simulation_run'].unique())}"
                    )
                if "reshuffle_run" in combined_df.columns:
                    print(
                        f"Reshuffle runs: {sorted(combined_df['reshuffle_run'].unique())}"
                    )
                if "fit_method" in combined_df.columns:
                    print(f"Fit methods: {sorted(combined_df['fit_method'].unique())}")
                if "experiment" in combined_df.columns:
                    print(f"Experiments: {sorted(combined_df['experiment'].unique())}")

                if "fit_method" in combined_df.columns:
                    print("\nCounts by fit_method:")
                    print(combined_df["fit_method"].value_counts())

                print("\nColumns:", list(combined_df.columns))

            # Save locally
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved combined results to: {output_file}")
            print(f"  Shape: {combined_df.shape}")

            # Upload to Azure if requested
            if upload_to_azure:
                print(f"\nUploading results to Azure datastore...")
                try:
                    # Upload to the same datastore, in a results subdirectory
                    upload_path = f"{dataset_path}/combined_results"
                    datastore.upload_files(
                        files=[str(output_file)],
                        target_path=upload_path,
                        overwrite=True,
                        show_progress=True,
                    )
                    print(
                        f"  ✓ Uploaded to: {datastore_name}/{upload_path}/{output_file.name}"
                    )
                except Exception as e:
                    print(f"  ✗ Error uploading to Azure: {e}")

    except Exception as e:
        print(f"  ✗ Error during mounting/processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("COMBINATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Mount Azure ML dataset and combine estimate results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with workspace config in current directory
  python combine_results_from_azure.py \\
      --datastore researcher_data \\
      --dataset_path AKK/experiments/trace/simulation/two_effects/bert/v01 \\
      --output combined_results.csv

  # With summary and upload back to Azure
  python combine_results_from_azure.py \\
      --datastore researcher_data \\
      --dataset_path AKK/experiments/trace/simulation/two_effects/bert/v01 \\
      --output combined_results.csv \\
      --print_summary \\
      --upload_to_azure

  # Filter to specific estimator
  python combine_results_from_azure.py \\
      --datastore researcher_data \\
      --dataset_path AKK/experiments/trace/simulation/two_effects/bert/v01 \\
      --output combined_bert_only.csv \\
      --estimator bert
        """,
    )

    # Azure ML arguments
    parser.add_argument(
        "--workspace_config",
        default=None,
        help="Path to Azure ML config.json file (default: use Workspace.from_config())",
    )
    parser.add_argument(
        "--datastore",
        required=True,
        help="Name of the Azure ML datastore (e.g., 'researcher_data')",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path within the datastore to results directory",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        default="combined_results.csv",
        help="Output CSV file path (default: combined_results.csv)",
    )

    # Filtering arguments
    parser.add_argument(
        "--estimator",
        nargs="*",
        choices=["baseline", "bert"],
        help="Filter to specific estimator(s) (default: include all)",
    )

    # Display arguments
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print summary statistics before saving",
    )

    # Azure upload argument
    parser.add_argument(
        "--upload_to_azure",
        action="store_true",
        help="Upload combined results back to Azure datastore",
    )

    args = parser.parse_args()

    # Run the mounting and combination
    mount_and_combine(
        datastore_name=args.datastore,
        dataset_path=args.dataset_path,
        output_path=args.output,
        workspace_config=args.workspace_config,
        estimators=args.estimator,
        print_summary=args.print_summary,
        upload_to_azure=args.upload_to_azure,
    )


if __name__ == "__main__":
    main()
