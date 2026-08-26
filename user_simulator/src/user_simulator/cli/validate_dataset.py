from pathlib import Path

import typer
from rich.console import Console

from user_simulator.data.loader import DatasetLoader
from user_simulator.data.validator import DatasetValidator
from user_simulator.exceptions import DatasetValidationError

app = typer.Typer(add_completion=False, help="Validate a reason-DAG JSONL dataset.")
console = Console()


@app.command()
def main(
    dataset: Path = typer.Option(
        Path("dataset/DAG.jsonl"), "--dataset", help="Path to the source JSONL dataset."
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat every validation warning as an error.",
    ),
    show_warnings: bool = typer.Option(
        False, "--show-warnings", help="Print non-fatal validation warnings."
    ),
) -> None:
    try:
        samples = DatasetLoader(dataset).load_all_samples()
        report = DatasetValidator().validate_samples(samples)
        report.raise_for_errors()
        if strict and report.warnings:
            detail = "\n".join(f"{issue.sample_id}: {issue.message}" for issue in report.warnings)
            raise DatasetValidationError(detail)
    except DatasetValidationError as exc:
        console.print(f"[red]Dataset validation failed:[/red]\n{exc}")
        raise typer.Exit(1) from exc
    if show_warnings:
        for warning in report.warnings:
            console.print(f"[yellow]warning[/yellow] {warning.sample_id}: {warning.message}")
    console.print(
        f"[green]Valid dataset:[/green] {len(samples)} samples, {len(report.warnings)} warning(s)"
    )


if __name__ == "__main__":
    app()
