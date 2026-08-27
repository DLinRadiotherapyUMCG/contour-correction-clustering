""" Step 2: generate contour alternatives """

import argparse
import sys
from pathlib import Path

# Make the downloaded repository runnable without installing it as a package.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_io.config import load_config
from workflows.generate import generate_contour_alternatives

def main(config_path: Path | None = None) -> int:
    """Run generation using a configuration file."""
    if config_path is None:
        parser = argparse.ArgumentParser(description="Generate contour alternatives")
        parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "generation.local.json", help="Path to the JSON configuration file")
        config_path = parser.parse_args().config
    if not config_path.is_file():
        raise FileNotFoundError(f"Create a local configuration file first: {config_path}")
    generate_contour_alternatives(load_config(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

