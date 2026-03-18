import pickle as pkl
import pandas as pd
import json
from pathlib import Path

def predict(input_data : str, model_path : str = None, output : str = "csv") -> str:
    if not model_path:
        raise ValueError("No model provided.")
    if not (input_data.endswith(".json") or input_data.endswith(".csv")):
        raise ValueError("Invalid prediction input file format. Please provide a JSON or CSV file.")
    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid model file format. Please provide a .pkl file.")
    try:
        with open(model_path, "rb") as f:
            model_data = pkl.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if input_data.endswith(".csv"):
        df = pd.read_csv(input_data)
    else:
        with open(input_data, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Invalid JSON format. Please provide a JSON object or an array of JSON objects.")



    expected = model_data.get("features")
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    predict_data = df[expected]
    prediction = model_data["model"].predict(predict_data)

    naming_data = model_data.get("naming")
    timestamp = naming_data.get("timestamp")
    model_type = naming_data.get("model_type")
    input_name = Path(input_data).stem
    output_path = f"artifacts/predictions/{model_type}_{timestamp}_{input_name}_preds"

    if not output:
        output = "csv"
    output = output.lower()
    if output not in ("csv", "json"):
        raise ValueError("Invalid output file format. Please write 'JSON' or 'CSV'.")
    output_df = df.copy()
    output_df["PredictedMedHouseVal"] = prediction
    if output == "csv":
        output_path += ".csv"
        output_df.to_csv(output_path, index=False)
    else:
        output_path += ".json"
        output_df.to_json(output_path, orient="records", indent=2)
        
    return output_path