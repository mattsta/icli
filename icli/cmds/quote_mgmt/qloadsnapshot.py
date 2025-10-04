"""Command: qloadsnapshot

Category: Quote Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["qloadsnapshot"])
@dataclass
class IOpQuoteLoadSnapshot(IOp):
    otherClientId: int = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*otherClientId",
                convert=lambda x: int(x) if x else self.state.clientId,
                desc="Optional Client ID if loading another client's snapshot",
            )
        ]

    async def run(self):
        cacheKey = ("quotes", f"client-{self.otherClientId}")
        cons = self.cache.get(cacheKey)  # type: ignore
        if not cons:
            # if no ids, just don't do anything.
            # also don't bother with any status/warning message because this runs on startup
            # and we don't need to know if we didn't restore anything.
            return False

        try:
            # snapshots are always saved with exact Contract objects, so we can just restore them directly
            cs = cons.get("contracts", [])
            for c in cs:
                self.state.addQuoteFromContract(c)

            logger.info("Restored {} quotes from snapshot", len(cs))

            return True
        except:
            # logger.exception("Failed?")
            # format is incorrect, just ignore it
            pass

        return False
