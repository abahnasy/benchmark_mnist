from pathlib import Path

import mlflow

if __name__ == "__main__":

    # Set tracking URI
    MODEL_REGISTRY = Path("experiments")
    if not Path(MODEL_REGISTRY).exists():
        raise Exception("Model registry does not exist")
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

    # Load all runs from experiment
    experiment_id = mlflow.get_experiment_by_name("baselines").experiment_id
    all_runs = mlflow.search_runs(experiment_ids=experiment_id, order_by=["metrics.val_loss ASC"])
    print(all_runs)
