"""Command: unset

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["unset"])
@dataclass
class IOpUnSetEnvironment(IOp):
    """Remove an environment variable (if set)."""

    key: str = field(init=False)

    def argmap(self):
        return [DArg("key", default="")]

    async def run(self):
        if not self.key:
            # if no input, just print current state
            self.state.updateGlobalStateVariable("", None)
            return

        self.state.updateGlobalStateVariable(self.key, "")
