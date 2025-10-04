"""Command: set

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["set"])
@dataclass
class IOpSetEnvironment(IOp):
    """Read or Write a global environment setting for this current client session.

    For a list of settable options, just run `set show`.
    To view the current value of an option, run `set [key]` with no value.
    To delete a key, use an empty value for the argument as `set [key] ""`.
    """

    key: str = field(init=False)
    val: list[str] = field(init=False)

    def argmap(self):
        return [DArg("key", default=""), DArg("*val", default="")]

    async def run(self):
        if not (self.key or self.val):
            # if no input, just print current state
            self.state.updateGlobalStateVariable("", None)
            return

        val = self.val[0] if self.val else None

        self.state.updateGlobalStateVariable(self.key, val)
