"""Base classes and decorators for the command system.

This module provides:
- IOp: Base class for all command operations
- @command: Decorator to register commands with metadata
- Command registry for auto-discovery
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mutil.dispatch

if TYPE_CHECKING:
    import icli.cli as typecheckingicli

# Global command registry - commands register themselves at import time
_COMMAND_REGISTRY: list[type["IOp"]] = []


def command(names: list[str] | str, category: str | None = None):
    """Decorator to register a command with metadata.

    The category is automatically inferred from the module's package hierarchy.
    Each category package (e.g., icli.cmds.portfolio) should define a CATEGORY
    constant in its __init__.py.

    Args:
        names: Command name(s) - either a single string or list of aliases
        category: Optional category override (auto-detected from module if not provided)

    Example:
        @command(names=["positions", "ls"])
        @dataclass
        class IOpPositions(IOp):
            ...
    """
    if isinstance(names, str):
        names = [names]

    def decorator(cls):
        # Auto-detect category from module path if not explicitly provided
        if category is None:
            # Get the module where the class is defined
            import importlib

            module_name = cls.__module__  # e.g., "icli.cmds.portfolio.positions"

            # Extract package name (e.g., "icli.cmds.portfolio")
            parts = module_name.split(".")
            if len(parts) >= 3 and parts[0] == "icli" and parts[1] == "cmds":
                category_package = ".".join(parts[:3])  # icli.cmds.portfolio

                # Import the category package to get CATEGORY constant
                try:
                    category_module = importlib.import_module(category_package)
                    detected_category = getattr(category_module, "CATEGORY", None)

                    if detected_category is None:
                        raise ValueError(
                            f"Category package {category_package} must define CATEGORY constant"
                        )

                    cls.__command_category__ = detected_category
                except ImportError:
                    raise ValueError(
                        f"Cannot import category package {category_package} for {cls.__name__}"
                    )
            else:
                raise ValueError(
                    f"Command {cls.__name__} must be in icli.cmds.<category> package structure"
                )
        else:
            cls.__command_category__ = category

        cls.__command_names__ = names
        _COMMAND_REGISTRY.append(cls)
        return cls

    return decorator


@dataclass
class IOp(mutil.dispatch.Op):
    """Common base class for all command operations.

    Provides shared functionality and state access for all commands.

    All commands must:
    - Inherit from this class
    - Be decorated with @command() to register
    - Implement argmap() to define arguments
    - Implement async run() to execute the command
    """

    # Note: this is a quoted annotation so python ignores but mypy can still use it
    state: "typecheckingicli.IBKRCmdlineApp"

    def __post_init__(self):
        """Initialize command with state shortcuts for convenience."""
        # for ease of use, populate state IB into our own instance
        assert self.state
        self.ib = self.state.ib
        self.cache = self.state.cache

    def runoplive(self, cmd, args=""):
        """Execute another command from within this command.

        Wrapper for delegating to other commands, e.g.:
            strikes = await self.runoplive("chains", self.symbol)
        """
        return self.state.dispatch.runop(cmd, args, self.state.opstate)

    def task_create(self, *args, **kwargs):
        """Create a background task."""
        return self.state.task_create(*args, **kwargs)
