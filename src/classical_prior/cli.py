import typer
from pathlib import Path

def main(
    source: str = typer.Argument(..., help="Folder of images, single image, video file, or 'camera:0'"),
    cfg: Path = typer.Option(Path("configs/classical_prior.yaml"), help="Path to YAML config"),
    display: bool = typer.Option(False, help="Preview frames in a window (testing only)"),
):
    from common.io import load_yaml, ConfigError
    from .pipeline import run

    try:
        config = load_yaml(cfg)
    except ConfigError as e:
        typer.echo(f"{e}")
        raise typer.Exit(2)

    try:
        frames = run(source=source, cfg=config, display=display)
    except Exception as e:
        typer.echo(f"Pipeline error: {e}")
        raise typer.Exit(3)

    typer.echo(f"Done. Frames are processed.")

def cli_entry():
    typer.run(main)

if __name__ == "__main__":
    cli_entry()