"""
Configuration module for ResearchLab.
Loads config.yaml and provides CONFIG object.
"""

import yaml
from pathlib import Path

# Get the directory where this file is located
config_dir = Path(__file__).parent

# Path to config.yaml
config_file = config_dir / "config.yaml"

# Load configuration
if config_file.exists():
    with open(config_file, 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"Configuration file not found: {config_file}")

# Export CONFIG for import
__all__ = ['CONFIG']
