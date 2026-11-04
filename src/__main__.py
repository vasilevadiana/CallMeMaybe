import sys

from src.callmemaybe import test_llm


def main() -> int:
    """Run the Call Me Maybe application."""
    try:
        test_llm()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
