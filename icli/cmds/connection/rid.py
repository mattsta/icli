"""Command: rid

Category: Connection
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["rid"])
@dataclass
class IOpRID(IOp):
    """Retrieve ib_insync request ID and server Next Request ID"""

    def argmap(self):
        # rid has no args!
        return []

    async def run(self):
        logger.info("CLI Request ID: {}", self.ib.client._reqIdSeq)
        logger.info(
            "Server Next Request ID: {} (see server log)", self.ib.client.reqIds(0)
        )
