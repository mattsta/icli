"""Command: qquote

Category: Live Market Quotes
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio

import pandas as pd


@command(names=["qquote"])
@dataclass
class IOpQQuote(IOp):
    """Quick Quote: Run a temporary quote request then print results when volatility is populated."""

    symbols: list[str] = field(init=False)

    def argmap(self) -> list[DArg]:
        return [DArg("*symbols")]

    async def run(self):
        if not self.symbols:
            logger.error("No symbols requested?")
            return

        contracts = [contractForName(sym) for sym in self.symbols]
        contracts = await self.state.qualify(*contracts)

        if not all(c.conId for c in contracts):
            logger.error("Not all contracts reported successful lookup!")
            logger.error(contracts)
            return

        # IBKR populates each quote data field async, so even after we
        # "request market data," it can take 5-10 seconds for all the fields
        # to become populated (if they even populate at all).
        tickers = []
        logger.info(
            "Requesting tickers for {}",
            ", ".join([c.localSymbol.replace(" ", "") or c.symbol for c in contracts]),
        )

        # TODO: check if we are subscribed to live quotes already and use live quotes
        #       instead of re-subscribing (also note to _not_ unsubscribe from already-existing
        #       live quotes if we merge them into the tickers check here too).
        for contract in contracts:
            # Request quotes with metadata fields populated
            # (note: metadata is only populated using "live" endpoints,
            #  so we can't use the self-canceling "11 second snapshot" parameter)
            tf = tickFieldsForContract(contract)
            # logger.info("[{}] Tick Fields: {}", contract, tf)
            tickers.append(self.ib.reqMktData(contract, tf))

        ATTEMPT_LIMIT = 10
        for i in range(ATTEMPT_LIMIT):
            ivhv = [
                all(
                    [
                        t.impliedVolatility,
                        t.histVolatility,
                        t.shortable,
                        t.shortableShares,
                    ]
                )
                for t in tickers
            ]

            # if any iv/hv are all populated, we have the data we want.
            if all(ivhv):
                break

            logger.warning(
                "Waiting for data to arrive... (attempt {} of {})",
                i,
                ATTEMPT_LIMIT,
            )
            await asyncio.sleep(1.33)
        else:
            logger.error("All data didn't arrive. Reporting partial results.")

        # logger.info("Got tickers: {}", pp.pformat(tickers))

        df = pd.DataFrame(tickers)

        # extract contract data from nested object pandas would otherwise
        # just convert to a blob of json text.
        contractframe = pd.DataFrame([t.contract for t in tickers])
        contractseries = contractframe["symbol secType conId".split()]

        # NB: 'halted' statuses are:
        # -1 Halted status not available.
        # 0 Not halted.
        # 1 General halt. regulatory reasons.
        # 2 Volatility halt.
        dfSlice = df[
            """bid bidSize
               ask askSize
               last lastSize
               volume open high low close vwap
               halted shortable shortableShares
               histVolatility impliedVolatility""".split()
        ]

        # attach inner name data to data rows since it's a nested field thing
        # this 'concat' works because the row index ids match across the contracts
        # and the regular ticks we extracted.
        dfConcat = pd.concat([contractseries, dfSlice], axis=1)

        printFrame(dfConcat)

        # all done!
        for contract in contracts:
            self.ib.cancelMktData(contract)
