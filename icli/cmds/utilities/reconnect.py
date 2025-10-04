"""Command: reconnect

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["reconnect"])
@dataclass
class IOpReconnect(IOp):
    """Run a full gateway reconnect cycle (primarily for debugging)"""

    shutdown: bool = field(init=False)

    def argmap(self):
        return [
            DArg(
                "shutdown",
                default=False,
                convert=lambda x: x.lower()
                in {"stop", "true", "shutdown", "goodbye", "bye"},
                desc="Whether to disconnect but NOT reconnect",
            )
        ]

    async def run(self):
        # Use the event framework to trigger the reconnect event handler.

        # if true, we disconnect then DO NOT RECONNECT
        # (note: we have no "connect" command, so this will leave your client in a
        #        'disconnected idle' state until you fully exit and restart)
        if self.shutdown:
            self.state.exiting = True

        got = self.state.ib.disconnect()
        logger.warning("{}", got)
