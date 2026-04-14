"""Command-line interface for ezQMMM 2.0."""

import sys
import yaml
import traceback

from ezqmmm.config import create_example_config
from ezqmmm.generator import QMMMGenerator


LOGO = r"""
--------------------------------------------------------
               ___  __  __ __  __ __  __   ____     ___
   ___  ____  / _ \|  \/  |  \/  |  \/  | |___ \   / _ \
  / _ \|_  / | | | | |\/| | |\/| | |\/| |   __) | | | | |
 |  __/ / /  | |_| | |  | | |  | | |  | |  / __/ _| |_| |
  \___|/___|  \__\_\_|  |_|_|  |_|_|  |_| |_____(_)\___/
             Easy QM/MM Input File Generator
                     Q-Chem · Orca 
--------------------------------------------------------
"""

def main():
    print(LOGO)
    if len(sys.argv) < 2:
        print("Usage:")
        print("  ezqmmm config.yaml")
        print("  ezqmmm --example")
        sys.exit(1)

    if sys.argv[1] == '--example':
        create_example_config()
        return

    try:
        with open(sys.argv[1]) as f:
            config = yaml.safe_load(f)
        gen = QMMMGenerator(config['psf_file'], config['dcd_file'])
        gen.generate(config)
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)
