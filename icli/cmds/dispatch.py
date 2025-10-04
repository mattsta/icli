"""Command dispatcher for routing commands to their handlers."""

from collections.abc import Coroutine
from dataclasses import dataclass

import mutil.dispatch


@dataclass
class Dispatch:
    """Dispatcher that routes command strings to their registered handlers.

    Uses the auto-discovered OP_MAP to find and execute commands.
    """

    def __post_init__(self):
        # Import OP_MAP here to avoid circular imports
        from . import OP_MAP

        self.d = mutil.dispatch.Dispatch(OP_MAP)

    def runop(self, *args, **kwargs) -> Coroutine:
        """Execute a command by name.

        Args:
            Command name and arguments (forwarded to mutil.dispatch)

        Returns:
            Coroutine that executes the command
        """
        return self.d.runop(*args, **kwargs)
