"""Re-check the shipped cases against the composition rules; rewrites benchmark/composite_ledger.json."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from vhhome_bench.case_checks import main

if __name__ == "__main__":
    _sys.exit(main())
