#!/usr/bin/env python3

import fire
import sys
import traceback
from pydantic import ValidationError
from src.commands import CLI


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except (ValidationError, Exception) as e:
        tb = sys.exc_info()[2]
        if tb is not None:
            details = traceback.extract_tb(tb)[-1]
            print(f"\nLine error: {details.lineno} in file {details.filename}")
        print(f"{e}\n")
