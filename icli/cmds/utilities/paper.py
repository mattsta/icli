"""Command: paper

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
import prettyprinter as pp  # type: ignore


@command(names=["paper"])
@dataclass
class IOpPaper(IOp):
    """Run a paper trading simulation of buying/selling instruments.

    Note: this is unrelated to any actual "paper trading" account and only exists for this single live session."""

    symbol: str = field(init=False)
    qty: float = field(init=False)
    price: float = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", desc="Symbol to trade"),
            DArg("qty", convert=float, desc="Quantity to trade"),
            DArg(
                "*price",
                convert=lambda x: float(x[0]) if x else None,
                desc="Price to trade (optional; will be looked up live if not provided)",
            ),
        ]

    async def run(self):
        symbol = self.symbol
        qty = self.qty
        price = self.price

        pls = self.state.paperLog[symbol]

        ssymbol, contract = await self.state.positionalQuoteRepopulate(self.symbol)
        if contract:
            quoteKey = lookupKey(contract)
            bid, ask = self.state.currentQuote(quoteKey, False)
            currentPrice = (bid + ask) / 2 if bid and ask else None
        else:
            currentPrice = None

        if qty != 0:
            pls.log(qty, price or currentPrice)  # type: ignore

        report = pls.report(currentPrice)
        logger.info("[{}] Profit Report: {}", symbol, pp.pformat(report))
