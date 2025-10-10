"""
Configuration generator for systematic experimentation.

This module generates combinations of configurations for thorough
experimental evaluation.
"""

from itertools import product
from pathlib import Path
from typing import Dict, List, Any
import yaml


class ConfigGenerator:
    """Generate experimental configurations."""

    @staticmethod
    def generate_combinations(base_config: Dict, param_grid: Dict) -> List[Dict]:
        """
        Generate all combinations of parameters.

        Args:
            base_config: Base configuration dictionary
            param_grid: Dictionary of parameters to combine
                Example:
                {
                    "optimization.method": ["vanilla", "fabo", "pls"],
                    "optimization.acquisition": ["ei", "pi", "ucb"],
                    "optimization.kernel": ["matern", "rbf"]
                }

        Returns:
            List of configuration dictionaries
        """
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        configs = []

        # Generate all combinations
        for values in product(*param_values):
            config = base_config.copy()

            # Update config with current combination
            for name, value in zip(param_names, values):
                # Handle nested parameters
                keys = name.split(".")
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value

            configs.append(config)

        return configs

    @staticmethod
    def save_configs(configs: List[Dict], output_dir: Path):
        """Save configurations to YAML files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, config in enumerate(configs):
            # Create descriptive filename
            parts = []
            if "method" in config.get("optimization", {}):
                parts.append(config["optimization"]["method"])
            if "acquisition" in config.get("optimization", {}):
                parts.append(config["optimization"]["acquisition"])
            if "kernel" in config.get("optimization", {}):
                parts.append(config["optimization"]["kernel"])

            filename = f"config_{'_'.join(parts)}_{i}.yaml"

            with open(output_dir / filename, "w") as f:
                yaml.dump(config, f, default_flow_style=False)


def generate_experiment_configs(
    base_config_path: str, output_dir: str, param_grid: Dict[str, List[Any]] = None
):
    """
    Generate experiment configurations.

    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to save generated configs
        param_grid: Parameter grid for combinations
    """
    # Load base configuration
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            "optimization.method": ["vanilla", "fabo", "pls", "pca", "opls"],
            "optimization.acquisition": ["ei", "pi", "ucb"],
            "optimization.kernel": ["matern", "rbf"],
        }

    # Generate configurations
    configs = ConfigGenerator.generate_combinations(base_config, param_grid)

    # Save configurations
    ConfigGenerator.save_configs(configs, Path(output_dir))

    return configs


# Example usage
if __name__ == "__main__":
    # Example parameter grid
    param_grid = {
        "optimization.method": ["vanilla", "fabo", "pls", "pca", "opls"],
        "optimization.acquisition": ["ei", "pi", "ucb"],
        "optimization.kernel": ["matern", "rbf"],
        "feature_selection.pls_components": [10, 20, 30],
    }

    configs = generate_experiment_configs(
        base_config_path="config/default_config.yaml",
        output_dir="config/experiments",
        param_grid=param_grid,
    )

    print(f"Generated {len(configs)} configurations")
