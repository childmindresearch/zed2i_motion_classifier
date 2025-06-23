"""CLI for zed2i_motion_classifier."""

import argparse
from typing import List, Optional

from core import orchestrator

def main() -> None:
    """Runs motion classification orchestrator with command line arguments.

    Args:
         args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--participant_id", required=True, type=str, help="Participant ID"
    )
    arguments = parser.parse_args()

    orchestrator.run(participant_id=arguments.participant_id)

