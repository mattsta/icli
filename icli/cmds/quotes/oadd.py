"""Command: oadd

Category: Live Market Quotes
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Contract,
)
from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["oadd"])
@dataclass
class IOpQuotesAddFromOrderId(IOp):
    """Add symbols for current orders to the quotes view."""

    orderIds: list[int] = field(init=False)

    def argmap(self):
        return [DArg("*orderIds", lambda xs: [int(x) for x in xs])]

    async def run(self):
        trades = self.ib.openTrades()

        if not self.orderIds:
            addTrades = trades
        else:
            addTrades = []
            for oid in self.orderIds:
                useTrade = None
                for t in trades:
                    if t.order.orderId == oid:
                        useTrade = t
                        break

                if not useTrade:
                    logger.error("No order found for id {}", oid)
                    continue

                addTrades.append(useTrade)

        # if we have no orders to add, don't do anything
        if not addTrades:
            return

        for useTrade in addTrades:
            # If this is a new session and we haven't previously cached the
            # contract id -> name mappings, we need to look them all up now
            # or else the next print of the quote toolbar will throw lots
            # of missing key exceptions when trying to find names.
            if useTrade.contract.comboLegs:
                # TODO: verify this logic still holds after the contract cache refactoring
                for x in useTrade.contract.comboLegs:
                    # if ID -> Name not in the cache, create it
                    if x.conId not in self.state.conIdCache:
                        await self.state.qualify(Contract(conId=x.conId))
            else:
                if useTrade.contract.conId not in self.state.conIdCache:
                    await self.state.qualify(Contract(conId=useTrade.contract.conId))

            self.state.addQuoteFromContract(useTrade.contract)

        # also update current client persistent quote snapshot
        await self.runoplive("qsnapshot")
