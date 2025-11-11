import typer
app = typer.Typer()

@app.command()
def run():
    print("CLI работает!")

if __name__ == "__main__":
    app()
