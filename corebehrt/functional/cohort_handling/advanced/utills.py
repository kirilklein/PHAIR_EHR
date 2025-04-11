import numpy as np
import re


def prettify_stats(stats: dict) -> dict:
    """
    Convert all numpy numeric types to Python native types recursively through the dictionary.

    Args:
        stats (dict): Dictionary containing statistics, potentially with numpy numeric types
                     and nested dictionaries

    Returns:
        dict: Dictionary with all numpy numeric types converted to Python native types
    """

    def convert_value(v):
        if isinstance(v, (np.integer, np.floating)):
            return int(v) if isinstance(v, np.integer) else float(v)
        elif isinstance(v, dict):
            return prettify_stats(v)
        elif isinstance(v, (list, tuple)):
            return type(v)(convert_value(x) for x in v)
        return v

    return {k: convert_value(v) for k, v in stats.items()}


def extract_criteria_names_from_expression(expression: str) -> list:
    """Extract criteria names from expression by splitting on operators and check they exist"""
    return [c.strip() for c in re.split(r"[|&~]", expression) if c.strip()]
