"""Command: say

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["say"])
@dataclass
class IOpSay(IOp):
    """Speak a custom phrase provided as arguments.

    Can be used as a standalone command or combined with scheduled events to create
    speakable events on a delay."""

    what: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*what")]

    async def run(self):
        content = " ".join(self.what)
        self.task_create(content, self.state.speak.say(say=content))
