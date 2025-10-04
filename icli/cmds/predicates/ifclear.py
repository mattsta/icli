"""Command: ifclear

Category: Predicate Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["ifclear"])
@dataclass
class IOpPredicateClearAll(IOp):
    """Delete ALL predicates immediately."""

    everything: bool = field(init=False)

    def argmap(self):
        return [
            DArg(
                "everything",
                default=False,
                desc="By default, only clear active predicates. If 'everything' is True, clear ALL cached predicates for all symbols",
            )
        ]

    async def run(self):
        if self.everything:
            logger.info("Deleting ALL Predicates")
            self.state.ifthenRuntime.clear()
        else:
            logger.info(
                "Stopping all predicates, but they still exist to be re-activated"
            )
            self.state.ifthenRuntime.clearActive()
