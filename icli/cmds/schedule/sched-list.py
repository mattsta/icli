"""Command: sched-list, slist

Category: Schedule Management
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["sched-list", "slist"])
@dataclass
class IOpScheduleEventList(IOp):
    """List scheduled events by name and command and target date."""

    async def run(self):
        logger.info("Listing {} scheduled events by name...", len(self.state.scheduler))
        self.state.scheduler.report()
