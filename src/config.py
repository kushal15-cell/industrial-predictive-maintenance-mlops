from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def load_params():
    params_path = ROOT_DIR / "params.yaml"

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    return params