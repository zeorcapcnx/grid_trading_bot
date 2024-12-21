import argparse, logging, os, traceback

def validate_args(args):
    """
    Validates parsed arguments.

    Args:
        args: Parsed arguments object.
    Raises:
        ValueError: If validation fails.
    """
    if args.save_performance_results:
        save_performance_dir = os.path.dirname(args.save_performance_results)
        if save_performance_dir and not os.path.exists(save_performance_dir):
            raise ValueError(f"The directory for saving performance results does not exist: {save_performance_dir}")

def parse_and_validate_console_args(cli_args=None):
    """
    Parses and validates console arguments.

    Args:
        cli_args: Optional CLI arguments for testing.
    Returns:
        argparse.Namespace: Parsed and validated arguments.
    Raises:
        RuntimeError: If argument parsing or validation fails.
    """
    try:
        parser = argparse.ArgumentParser(description="Spot Grid Trading Strategy.")
        parser.add_argument('--config', type=str, nargs='+', required=True, metavar='CONFIG', help='Path(s) to config file(s).')
        parser.add_argument('--save_performance_results', type=str, metavar='FILE', help='Path to save simulation results (e.g., results.json)')
        parser.add_argument('--no-plot', action='store_true', help='Disable the display of plots at the end of the simulation')
        parser.add_argument('--profile', action='store_true', help='Enable profiling')
        args = parser.parse_args(cli_args)
        validate_args(args)
        return args

    except SystemExit as e:
        logging.error(f"Argument parsing failed: {e}")
        raise RuntimeError("Failed to parse arguments. Please check your inputs.") from e
    
    except ValueError as e:
        logging.error(f"Validation failed: {e}")
        raise RuntimeError("Argument validation failed.") from e
    
    except Exception as e:
        logging.error(f"An unexpected error occurred while parsing arguments: {e}")
        logging.error(traceback.format_exc())
        raise RuntimeError("Unexpected error during argument parsing.") from e