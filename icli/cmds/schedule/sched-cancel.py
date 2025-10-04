"""Command: sched-cancel, scancel

Category: Schedule Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["sched-cancel", "scancel"])
@dataclass
class IOpScheduleEventCancel(IOp):
    """Cancel event by name."""

    name: str = field(init=False)

    def argmap(self):
        return [DArg("name", desc="Name of event to cancel")]

    async def run(self):
        got = self.state.scheduler.cancel(self.name)
        if not got:
            logger.error("[{}] Scheduled event not found?", self.name)
            return False

        logger.info("[{} :: {}] Command(s) deleted!", self.name, [g.meta for g in got])
        logger.info("[{} :: {}] Task(s) deleted!", self.name, got)
