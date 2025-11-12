"""
CLI entry for the 'classical+neural prior' pipeline.

Usage example:
  rdw-classical data/frames_2024 --display
"""

import typer
from pathlib import Path


def main(
    source: str = typer.Argument(..., help="Image file, folder, video file, or 'camera:0'"),
    cfg: Path = typer.Option(Path("configs/classical_prior.yaml"), help="Path to YAML config"),
    display: bool = typer.Option(False, help="Show mask and final result windows (testing)"),
):
    """
    Load config, run the pipeline on the given source, and report how many frames were processed.
    Exits with non-zero status on config or runtime errors.
    """
    from common.io import load_yaml, ConfigError
    from .pipeline import run

    try:
        config = load_yaml(cfg)
    except ConfigError as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)

    try:
        frames = run(source=source, cfg=config, display=display)
    except Exception as e:
        typer.echo(f"Pipeline error: {e}")
        raise typer.Exit(code=3)

    typer.echo("Done. Frames are processed.")


def cli_entry():
    typer.run(main)


if __name__ == "__main__":
    cli_entry()