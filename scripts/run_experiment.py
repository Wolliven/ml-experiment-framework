"""
Entrypoint for training the experiment.
Loads a dataset from a CSV file and an experiment YAML configuration file,
trains a model, evaluates it, saves the model artifact, and logs the experiment results.
"""
import sys
import argparse

def main() -> None:
    from ml_engine.training import train_model
    parser = argparse.ArgumentParser(
        description="Run a machine learning experiment based on the provided YAML configuration file.",
        epilog="Example: python -m scripts.run_experiment configs/experiment_001.yaml"
    )
    parser.add_argument("experiment", type=str, help="Path to the experiment YAML file.")
    args = parser.parse_args()

    result = train_model(config_path=args.experiment)
    model_path = result.get("model_path")
    if not model_path:
        raise ValueError("Training did not return a model_path.")
    print(f"Experiment: {args.experiment}")
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