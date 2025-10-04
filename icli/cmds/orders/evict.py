"""Command: evict

Category: Order Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    FuturesOption,
    Option,
)
from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio


@command(names=["evict"])
@dataclass
class IOpPositionEvict(IOp):
    """Evict a position using automatic MIDPRICE sell order for equity or ADAPTIVE FAST for options and futures.

    Note: the symbol name accepts '*' for wildcards!

    Also note: for futures, the actual symbol has the month expiration attached like "MESU2", so the portfolio
               symbol is not just "MES". Evicting futures reliably uses evict MES* and not MES or /MES.
    """

    sym: str = field(init=False)
    qty: float = field(init=False)
    delta: float = field(init=False)
    algo: str | None = field(init=False)

    def argmap(self):
        return [
            DArg("sym"),
            DArg(
                "qty",
                convert=float,
                verify=lambda x: x != 0 and x >= -1,
                default=-1,
                desc="qty is the exact quantity to evict (or -1 to evict entire position)",
            ),
            DArg(
                "delta",
                convert=float,
                verify=lambda x: 0 <= x <= 1,
                default=0,
                desc="only evict matching contracts with current delta >= X (not used if symbol isn't an option). deltas are positive for all contracts in this case (so asking for 0.80 will evict calls with delta >= 0.80 and puts with delta <= -0.80)",
            ),
            DArg(
                "*algo",
                desc="Optionally provide your own evict algo name to override the default choice",
            ),
        ]

    async def run(self):
        contracts = self.state.contractsForPosition(
            self.sym, None if self.qty == -1 else self.qty
        )

        if not contracts:
            logger.error("No contracts found for: {}", self.sym)
            return None

        runners = []
        for contract, qty, delayedEstimatedMarketPrice in contracts:
            # use a live midpoint market price as our initial offer
            quoteKey = lookupKey(contract)
            bid, ask = self.state.currentQuote(quoteKey)

            assert bid and ask
            price = (bid + ask) / 2

            if self.delta:
                # if asking for a delta eviction, check current quote...
                quotesym = lookupKey(contract)

                # verify quote is loaded...
                if not self.state.quoteExists(contract):
                    logger.info("Quote didn't exist, adding now...")
                    await self.runoplive(
                        "add",
                        f'"{quotesym}"',
                    )

                # check delta...
                while not (
                    thebigd := self.state.quoteState[quotesym].modelGreeks.delta
                ):
                    # takes a couple moments for the greeks feed to populate on initial quote...
                    await asyncio.sleep(0.003)

                # skip placing this contract order if the delta is below the user requested threshold.
                # (i.e. equivalent to "only evict if self.delta >= abs(contract delta)")
                if abs(thebigd) < self.delta:
                    continue

            # we only need to qualify if the ID doesn't exist
            if not contract.conId:
                (contract,) = await self.state.qualify(contract)

            algo = "MIDPRICE"

            # Note: we can't evict spreads/Bags because those must be constructed as multi-leg orders and
            #       our eviction logic has no way to discover what the user's intent would be.
            # TODO: when opening a spread, we should record the positions as a spread so we can flip sides for easier closing.
            if isinstance(contract, (Option, FuturesOption)):
                algo = "AF"
            elif isinstance(contract, Future):
                algo = "PRTMKT"

            # if user provided their own algo name, override all our defaults and use the user's algo choice instead
            if self.algo:
                algo = self.algo[0]

            logger.info(
                "[{}] [{}] Submitting through spread tracking order automation...",
                self.sym,
                (contract.localSymbol, qty, price),
            )

            # TODO: this isn't the most efficient because we are sending this to another
            #       full command parser instance, so it has to re-do some of our work again.
            #       GOAL: move the _entire_ "buy tracking" logic into placeOrderForContract() directly
            #             as an option, then call .placeOrderForContract() here again so we don't
            #             have to find everything again.

            # closing is the opposite of the quantity sign from the portfolio (10 long to -10 short (close), etc)
            qty = -qty
            runners.append(self.runoplive("buy", f"{quoteKey} {qty} {algo}"))

        if runners:
            await asyncio.gather(*runners)
