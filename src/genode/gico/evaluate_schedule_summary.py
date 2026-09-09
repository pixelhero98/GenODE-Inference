"""Validation reporting shares the same frozen reward construction as locked reporting."""

from genode.gico.report_locked_test import report_main


def main() -> None:
    report_main(default_split="validation")


if __name__ == "__main__":
    main()
