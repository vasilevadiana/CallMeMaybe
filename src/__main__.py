import argparse
import sys

from src.callmemaybe import test_llm


def main(argv: list[str] | None = None) -> int:
    """Run the Call Me Maybe application."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    args = parser.parse_args(argv)

    try:
        test_llm(args.functions_definition, args.input, args.output)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
