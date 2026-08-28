#!/usr/bin/env python3

import fire
from src.commands import CLI


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except (Exception) as e:
        print(e)
