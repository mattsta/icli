"""Command system with auto-discovery and registration.

This module provides:
- Auto-discovery of all commands from category subdirectories
- OP_MAP: The master command registry mapping categories and names to handlers
- Dispatch: The command dispatcher

Commands are automatically discovered by:
1. Scanning all .py files in category subdirectories
2. Importing them to trigger @command decorator registration
3. Building OP_MAP from the registry at module load time
"""

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Type

from .base import _COMMAND_REGISTRY, IOp, command
from .dispatch import Dispatch

__all__ = ["OP_MAP", "Dispatch", "IOp", "command"]


def discover_and_import_commands():
    """Discover and import all command modules from category directories.

    Scans all subdirectories of cmds/ for .py files (excluding __init__.py,
    base.py, dispatch.py, utils.py) and imports them to trigger @command registration.

    This is the single point of command discovery - no duplication needed.
    """
    cmds_dir = Path(__file__).parent

    # Files to skip (infrastructure, not commands)
    skip_files = {"__init__.py", "base.py", "dispatch.py", "utils.py"}

    # Find all category directories
    for category_dir in cmds_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue

        # Import all .py files in this category
        for module_file in category_dir.glob("*.py"):
            if module_file.name in skip_files:
                continue

            # Build module path like: icli.cmds.quotes.qquote
            module_name = f".{category_dir.name}.{module_file.stem}"
            importlib.import_module(module_name, package=__package__)


def build_op_map() -> Mapping[str, Mapping[str, type[IOp]]]:
    """Build OP_MAP from auto-discovered commands.

    Returns:
        Nested dict: {category: {command_name: CommandClass}}
    """
    # First, discover and import all commands
    discover_and_import_commands()

    # Build the OP_MAP from registered commands
    op_map: dict[str, dict[str, type[IOp]]] = {}

    # Track all command names globally to detect cross-category duplicates
    all_command_names: dict[str, tuple[str, type[IOp]]] = {}

    for cmd_class in _COMMAND_REGISTRY:
        category = cmd_class.__command_category__
        if category not in op_map:
            op_map[category] = {}

        # Register all command name aliases
        for name in cmd_class.__command_names__:
            # Check for duplicates within same category
            if name in op_map[category]:
                existing = op_map[category][name]
                raise ValueError(
                    f"Duplicate command name '{name}' in category '{category}': "
                    f"{existing.__name__} and {cmd_class.__name__}"
                )

            # Check for duplicates across all categories
            if name in all_command_names:
                existing_category, existing_class = all_command_names[name]
                raise ValueError(
                    f"Duplicate command name '{name}' across categories:\n"
                    f"  - {existing_category}: {existing_class.__name__}\n"
                    f"  - {category}: {cmd_class.__name__}"
                )

            all_command_names[name] = (category, cmd_class)
            op_map[category][name] = cmd_class

    return op_map


# Build the master command registry
OP_MAP = build_op_map()
