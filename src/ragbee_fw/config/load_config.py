"""Configuration loading utility for the RagBee framework.

This module provides a utility function to load an application configuration
from a YAML file into a validated Pydantic model.
"""

import yaml

from ragbee_fw.core.models.app_config import AppConfig


def load_config(path: str) -> AppConfig:
    """Load application configuration from a YAML file.

    This function reads a YAML file containing application settings, parses it
    into a dictionary, and validates it against the `AppConfig` Pydantic model.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        AppConfig: An instance of the application configuration with validated fields.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.
        pydantic.ValidationError: If the parsed data does not match the AppConfig model.
    """
    with open(path) as stream:
        config = yaml.safe_load(stream=stream)
    return AppConfig(**config)
