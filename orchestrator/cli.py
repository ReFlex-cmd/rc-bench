from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .prepare_data import prepare_narma10

# Это Typer-приложение, а НЕ функция
app = typer.Typer(help="RC-bench orchestrator CLI")
console = Console()


@app.command()
def prepare(
    task: str = typer.Option(
        "narma10",
        "--task",
        "-t",
        help="Имя задачи (пока поддерживается только narma10)",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        "-o",
        help="Каталог для сохранения подготовленных данных (по умолчанию data/prepared/<task>)",
    ),
    length: int = typer.Option(
        20000,
        "--length",
        "-l",
        help="Длина временного ряда (кол-во тактов)",
    ),
    train_frac: float = typer.Option(0.6, help="Доля train"),
    val_frac: float = typer.Option(0.2, help="Доля val (остальное пойдёт в test)"),
    seed: int = typer.Option(42, help="Сид генератора случайных чисел"),
):
    """
    Подготовить данные для выбранной задачи (сейчас: NARMA10).
    """
    if out_dir is None:
        out_dir = Path("data") / "prepared" / task

    console.rule(f"[bold cyan]Подготовка данных для задачи: {task}")
    console.print(f"[bold]Выходной каталог:[/bold] {out_dir}")

    task_lower = task.lower()

    if task_lower == "narma10":
        out_dir.mkdir(parents=True, exist_ok=True)
        prepare_narma10(
            out_dir=out_dir,
            length=length,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )
        console.print("[green]Готово:[/green] сгенерированы и сохранены данные NARMA10.")
    else:
        console.print(f"[red]Пока не поддерживаю задачу '{task}'.[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    # Локальный запуск: python orchestrator/cli.py prepare ...
    app()
