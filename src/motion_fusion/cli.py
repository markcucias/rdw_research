# src/motion_fusion/cli.py
import typer
from pathlib import Path
import yaml

from .pipeline import run


def main(
    source: str = typer.Argument(..., help="Image / folder / video / camera:0"),
    cfg: Path = typer.Option(Path("configs/motion_fusion.yaml"), help="YAML config"),
    display: bool = typer.Option(False, help="Show debug windows"),
):
    with cfg.open("r") as f:
        config = yaml.safe_load(f)

    frames = run(source=source, cfg=config, display=display)
    typer.echo(f"Done. Frames processed: {frames}")


def cli_entry():
    typer.run(main)


if __name__ == "__main__":
    cli_entry()
