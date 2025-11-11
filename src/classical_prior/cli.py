import typer
from pathlib import Path


# Define the Typer-powered function
def main(
    source: str = typer.Argument(..., help="Folder of images, single image, video file, or 'camera:0'"),
    cfg: Path = typer.Option(Path("configs/classical_prior.yaml"), help="Path to YAML config"),
    display: bool = typer.Option(False, help="Preview frames in a window (testing only)"),
):
    """
    Single-command CLI: reads frames from SOURCE and processes them headlessly by default.
    Use --display to preview during development; keep it headless for deployment.
    """
    
    from common.io import load_yaml, ConfigError
    try:
        config = load_yaml(cfg)
    except ConfigError as e:
        typer.echo(f"{e}")
        raise typer.Exit(2)

    # Checking if config loaded correctly
    top_keys = ", ".join(sorted(config.keys())) if config else "(empty)"
    typer.echo(f"Config loaded: {cfg}  keys: {top_keys}")
    
    from common.video import open_source 
    cap = open_source(source)

    if not cap.isOpened():
        typer.echo(f"Could not open source: {source}")
        raise typer.Exit(1)

    frames = 0
    last_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frames += 1
        last_frame = frame

        if display:
            import cv2
            cv2.imshow("RDW - Classical (preview)", frame)
            wait = 0 if getattr(cap, "single_image", False) else 1
            if cv2.waitKey(wait) & 0xFF == 27:
                break

    cap.release()

    if display and last_frame is not None:
        import cv2
        if not getattr(cap, "single_image", False):
            cv2.imshow("RDW - Classical (preview)", last_frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    typer.echo(f"Done. Frames processed: {frames}")


def cli_entry():
    typer.run(main)

if __name__ == "__main__":
    cli_entry()