"""Print the resolved config and selected device. The project's hello-world.

Usage:
    python scripts/show_config.py
    python scripts/show_config.py device=cpu run.name=smoke_test
    python scripts/show_config.py material.restitution=0.5
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from warp_dem import describe_device, resolve_device


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("\n--- Resolved config ---")
    print(OmegaConf.to_yaml(cfg))

    device = resolve_device(cfg.device)
    print("--- Device ---")
    print(describe_device(device))
    print(f"requested: {cfg.device!r}  ->  resolved: {device}\n")


if __name__ == "__main__":
    main()