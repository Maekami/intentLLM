from pathlib import Path

import typer
import uvicorn
from rich.console import Console

from interaction_pipeline.cli.common import configure
from interaction_pipeline.web import create_app

app = typer.Typer(add_completion=False, help="Serve a live full-reply interaction panel.")
console = Console()


@app.command()
def main(
    sample_id: str = typer.Option(..., "--sample-id", help="Exact dataset sample ID."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    difficulty: str | None = typer.Option(None, "--difficulty", help="easy, medium, or hard."),
    seed: int | None = typer.Option(None, "--seed"),
    baseline: str | None = typer.Option(None, "--baseline", help="base or prompt_base."),
    assistant_model_profile: str | None = typer.Option(
        None, "--assistant-model-profile", help="Assistant model profile name or YAML path."
    ),
    simulator_model_profile: str | None = typer.Option(
        None, "--simulator-model-profile", help="Simulator model profile name or YAML path."
    ),
    max_turns: int | None = typer.Option(None, "--max-turns", min=1),
    pipeline_config: Path | None = typer.Option(None, "--pipeline-config"),
    simulator_config: Path | None = typer.Option(None, "--simulator-config"),
    assistant_config: Path | None = typer.Option(None, "--assistant-config"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    _, prepared, resolved_output = configure(
        pipeline_config=pipeline_config,
        simulator_config=simulator_config,
        assistant_config=assistant_config,
        simulator_model_profile=simulator_model_profile,
        assistant_model_profile=assistant_model_profile,
        baseline=baseline,
        difficulty=difficulty,
        seed=seed,
        concurrency=None,
        sample_retries=None,
        output_dir=output_dir,
        max_turns=max_turns,
    )
    if prepared.missing_credentials:
        console.print(
            "[red]Missing required credentials: "
            + ", ".join(prepared.missing_credentials)
            + ".[/red]"
        )
        raise typer.Exit(2)
    # Validate before starting the server so bad IDs fail immediately.
    prepared.dataset.load_sample_by_id(sample_id)
    web_app = create_app(prepared, sample_id=sample_id, output_root=resolved_output)
    console.print(f"Panel: http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
