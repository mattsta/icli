"""Command: qsnapshot

Category: Quote Management
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["qsnapshot"])
@dataclass
class IOpQuoteSaveClientSnapshot(IOp):
    """Save the current live quote state for THIS CLIENT ID only so it auto-reloads on the next startup."""

    def argmap(self):
        return []

    async def run(self):
        cacheKey = ("quotes", f"client-{self.state.clientId}")
        allLiveContracts = [c.contract for c in self.state.quoteState.values()]
        self.cache.set(cacheKey, {"contracts": allLiveContracts})  # type: ignore

        # This log line is nice for debugging but too noisy to run on every snapshot
        # since every 'add' or 'oadd' is a new snapshot saving event too.
        if False:
            logger.info(
                "[{}] Saved {} contract ids for snapshot: {}",
                cacheKey,
                len(allLiveContracts),
                sorted(
                    [
                        (c.contract.localSymbol, c.contract.conId)
                        for c in self.state.quoteState.values()
                    ]
                ),
            )
