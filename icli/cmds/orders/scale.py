"""Command: scale

Category: Order Management
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["scale"])
@dataclass
class IOpScaleOrder(IOp):
    """Scale-in order entry where you specify an instrument and a starting time, and it grabs prices in ±tick increments until quantity filled."""

    cmd: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*cmd",
                desc="Command in Order Lang format for buying or selling or previewing operations",
            )
        ]

    async def run(self):
        """Begin the scale-in order process for the given order command request."""
        cmd = " ".join([f"'{c}'" if " " in c else c for c in self.cmd])

        # parse the entire input string to this command through the requestlang/orderlang parser
        request = self.state.requestlang.parse(cmd)
        logger.info("[{}] Requesting: {}", cmd, request)

        self.symbol = request.symbol
        assert self.symbol
        assert request.qty

        contract = None
        self.symbol, contract = await self.state.positionalQuoteRepopulate(self.symbol)

        if not contract:
            logger.error("Contract not found for: {}", self.symbol)
            return None

        isPreview: Final = request.preview

        name = nameForContract(contract)
        tickLow = request.config.get("ticklow", 0.25)
        tickHigh = request.config.get("tickhigh", 0.25)

        assert isinstance(tickLow, (float, Decimal))
        assert isinstance(tickHigh, (float, Decimal))

        self.state.task_create(
            f"[{name}] Scale-In Automation",
            self.state.positionActiveLifecycleDoctrine(
                contract, request, tickLow, tickHigh
            ),
        )
