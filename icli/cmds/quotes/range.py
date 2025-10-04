"""Command: range

Category: Live Market Quotes
"""

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio

import numpy as np
import whenever


@command(names=["range"])
@dataclass
class IOpOptionChainRange(IOp):
    """Print OCC symbols for an underlying with ± offset for puts and calls."""

    symbol: str = field(init=False)
    width: float = field(init=False)
    rename: str | None = field(init=False)

    def argmap(self):
        # TODO: add options to allow multiple symbols and also printing cached chains
        return [
            DArg("symbol", convert=lambda x: x.upper()),
            DArg("width", convert=lambda x: float(x)),
            DArg(
                "*rename",
                convert=lambda x: x[0].upper() if x else None,
                desc="Optionally rename the generated strike with a different symbol name (e.g. 'range SPX 20 SPXW')",
            ),
        ]

    async def run(self):
        # verify we have quotes enabled to get the current price...
        await self.runoplive("add", f'"{self.symbol}"')

        datestrikes = await self.runoplive(
            "chains",
            self.symbol,
        )

        # TODO: what about ranges where we do'nt have an underlying like NDXP doesn't mean we are subscribed to ^NDX
        quote = self.state.quoteState[self.symbol]

        # prices are NaN until they get populated...
        # (and sometimes this doesn't work right after hours or weekend times... thanks ibkr)
        # TODO: we could _also_ check the bid/ask spread if the prices aren't populating.
        while (quote.last != quote.last) or (quote.close != quote.close):
            logger.info("[{}] Waiting for price to populate...", self.symbol)
            await asyncio.sleep(0.25)

        # this is a weird way of masking out all NaN values leaving only the good values, so we select
        # the first non-NaN value in this order here...
        fetchone = np.array([quote.last, quote.close])

        # mypy broke numpy resolution for some methods. mypy thinks `isfinite` doesn't exist when it clearly does.
        currentPrice = fetchone[np.isfinite(fetchone)][0]  # type: ignore

        low = currentPrice - self.width
        high = currentPrice + self.width
        logger.info(
            "[{} :: {}] Providing option chain range between [{}, {}] using current price {:,.2f}",
            self.symbol,
            self.width,
            low,
            high,
            currentPrice,
        )

        # TODO: for same-day expiry, we should be checking if the time of 'now' is >= (16, 15) and then use the next chain date instead.
        # Currently, if you request 'range' for daily options at like 5pm, it still gives you the same-day already expired options.
        now = whenever.ZonedDateTime.now("US/Eastern")
        today = str(now.date()).replace("-", "")
        generated = []

        # we need to sort by the date values since the IBKR API returns dates in a random order...
        for date, strikes in sorted(
            datestrikes[self.symbol].items(), key=lambda x: x[0]
        ):
            # don't generate strikes in the past...
            if date < today:
                logger.warning("[{}] Skipping date because it's in the past...", date)
                continue

            logger.info("[{}] Using date as basis for strike generation...", date)
            for strike in strikes:
                if low < strike < high:
                    for pc in ("P", "C"):
                        generated.append(
                            f"{self.rename or self.symbol}{date[2:]}{pc}{int(strike * 1000):0>8}"
                        )

            # only print for THE FIRST DATE FOUND
            # (so we don't end up writing these out for every daily expiration or
            #  an entire month of expirations, etc)
            break

        for row in generated:
            logger.info("Got: {}", row)

        out = f"strikes-{self.rename or self.symbol}.txt"
        pathlib.Path(out).write_text("\n".join(generated) + "\n")

        logger.info("[{} :: {}] Saved strikes to: {}", self.symbol, self.width, out)
