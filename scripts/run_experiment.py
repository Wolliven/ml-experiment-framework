"""
Entrypoint for training the experiment.
This script loads a dataset from a CSV file and a experiment yaml file, preprocesses the data,
trains a model, evaluates its performance, saves
the trained model to disk and saves the experiment results to a JSON file.
"""
import sys
import argparse

def main() -> None:
    from ml_engine.training import train_model
    parser = argparse.ArgumentParser(
        description="Run a machine learning experiment based on the provided YAML configuration file.",
        epilog="Example: python scripts/run_experiment.py configs/experiment_001.yaml"
    )
    parser.add_argument("experiment", type=str, help="Path to the experiment YAML file.")
    args = parser.parse_args()

    result = train_model(experiment_path=args.experiment)
    model_path = result.get("model_path")
    print(f"Trained model saved to: {model_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting.")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)