import optuna
import pandas as pd
import json
import argparse
import os


def extract_results(study_name, storage_url, output_dir):
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    os.makedirs(output_dir, exist_ok=True)

    if study.best_trial:
        best_params_path = os.path.join(output_dir, f"{study_name}_best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(study.best_trial.params, f, indent=4)
        print(f"Best parameters saved to: {best_params_path}")

        # 打印最佳参数和值
        print("\n--- Best Trial ---")
        print(f"  Trial number: {study.best_trial.number}")
        print(f"  Value (Objective): {study.best_trial.value}")
        print("  Parameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

    trials_df_path = os.path.join(output_dir, f"{study_name}_all_trials.csv")
    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_df_path, index=False)
    print(f"All trials data saved to: {trials_df_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Optuna Study Results")
    parser.add_argument("--study_name", required=True, help="Name of the Optuna study.")
    parser.add_argument("--storage_url", required=True, help="Optuna storage URL (e.g., sqlite:///./path/to/study.db)")
    parser.add_argument("--output_dir", default="./optuna_analysis_results", help="Directory to save extracted results.")

    args = parser.parse_args()
    extract_results(args.study_name, args.storage_url, args.output_dir)
