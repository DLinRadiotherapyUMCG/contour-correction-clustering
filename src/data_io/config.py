import json
from pathlib import Path
from typing import Any


def load_config(path: Path | str):
    """
    Load a JSON configuration file and convert configured paths to Path objects.
    
    Args:
        path (Path | str): Path to the JSON configuration file.

    Returns:
        dict: Configuration dictionary with Path objects for file paths.

    """

    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)

    paths = config.setdefault("paths", {})

    for name in ("ct", "ground_truth", "prediction", "alternatives", "tuning_results", "tuning_output"):
        if paths.get(name) is not None:
            paths[name] = Path(paths[name])

    return config

