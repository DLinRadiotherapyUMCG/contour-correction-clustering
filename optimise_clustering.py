""" Step 1: optimise clustering parameters """

import argparse
import sys
from pathlib import Path

# Make the downloaded repository runnable without installing it as a package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_io.config import load_config
from workflows.optimise import optimise_clustering


def main(config_path: Path | None = None) -> int:
    """Run optimisation using a configuration file."""
    if config_path is None:
        parser = argparse.ArgumentParser(description="Optimise contour clustering parameters")
        parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "generation.local.json", help="Path to the JSON configuration file")
        config_path = parser.parse_args().config

    config = load_config(config_path)
    clustering_algorithm = config.get("clustering_algorithm", "HDBSCAN")

    optimise_clustering(config, clustering_algorithm=clustering_algorithm)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
