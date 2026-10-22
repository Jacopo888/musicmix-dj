"""CLI for an explicit, hashable join of listening feedback to one run key."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from musicmix.evaluation import (  # noqa: E402
    EvaluationError,
    _run_root_and_key,
    import_judgements,
    render_report,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Importa un export musicmix.listening.1 e lo associa a un singolo run "
            "tramite listening_ab/key.json."
        )
    )
    parser.add_argument(
        "--judgements",
        required=True,
        type=Path,
        help="JSON dei giudizi esportato dalla pagina A/B",
    )
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="directory del run oppure il suo listening_ab/key.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON normalizzato; default: accanto all'input con nome esplicito",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="facoltativo: report Markdown per questo import",
    )
    parser.add_argument(
        "--mapping-confirmed",
        action="store_true",
        help="usa i nomi di sistema come mapping confermato dall'utente",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = import_judgements(
            args.judgements,
            args.run,
            mapping_confirmed=args.mapping_confirmed,
        )
        run_root, _ = _run_root_and_key(args.run)
        output = args.output
        if output is None:
            output = (
                Path(args.judgements).expanduser().resolve().parent
                / f"feedback_{Path(args.judgements).stem}_{run_root.name}.json"
            )
        output = write_json(document, output)
        report_path = None
        if args.report is not None:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_report(document), encoding="utf-8")
        # Keep terminal output metadata-only: comments are user evidence and
        # may contain lyrics or other private text.
        print(
            f"imported {document['summary']['observation_count']} observations; "
            f"mapping={document['mapping']['status']}; output={output}"
        )
        if report_path is not None:
            print(f"report={report_path}")
        return 0
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
