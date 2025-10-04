"""Command: tasklist

Category: Task Management
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["tasklist"])
@dataclass
class IOpTaskList(IOp):
    def argmap(self):
        return []

    async def run(self):
        self.state.task_report()
