"""CLI for zed2i_motion_classifier."""

import argparse
from typing import List, Optional

from zed2i_motion_classifier.core import orchestrator


def parse_arguments(args: Optional[List[str]]) -> argparse.Namespace:
    """Argument parser for zed2i_motion_classifier cli.

    Args:
        args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        Namespace object with all the input arguments and default values.

    Raises:
        SystemExit: if arguments are None.
    """
    parser = argparse.ArgumentParser(
        description="Run the main moiton classification pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Please report issues at https://github.com/childmindresearch/zed2i_motion_classifier.",
    )
    parser.add_argument(
        "-p",
        "--participant_id",
        type=str,
        required=True,
        help="String containing the participant's ID number.",
    )

    return parser.parse_args(args)


def main(
    args: Optional[List[str]] = None,
) -> None:
    """Runs motion classification orchestrator with command line arguments.

    Args:
         args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    """
    arguments = parse_arguments(args)

    orchestrator.run(participant_id=arguments.participant_id)
