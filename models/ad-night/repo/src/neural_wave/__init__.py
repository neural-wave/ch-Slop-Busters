"""Shield-first, code-aware anonymization of patient-doctor encounters (Neural Wave x Bosch)."""

__version__ = "0.2.0"


def main() -> int:
    from .cli import main as _main

    return _main()
