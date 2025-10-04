"""Command: depth

Category: Live Market Quotes
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from ib_async import (
    Bag,
)
from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio

import pandas as pd


@command(names=["depth"])
@dataclass
class IOpDepth(IOp):
    sym: str = field(init=False)
    count: int = 3

    def argmap(self):
        return [
            DArg("sym"),
            DArg(
                "*count",
                convert=lambda x: int(x[0]) if x else 3,
                verify=lambda x: 0 < x < 300,
                desc="depth checking iterations should be more than zero and less than a lot",
            ),
        ]

    async def run(self):
        self.sym: str
        try:
            foundsym, contract = await self.state.positionalQuoteRepopulate(self.sym)
            assert foundsym and contract

            self.sym = foundsym
        except Exception as e:
            logger.error("No contract found for: {} ({})", self.sym, str(e))
            return

        # logger.info("Available depth: {}", await self.ib.reqMktDepthExchangesAsync())

        self.depthState = {}
        useSmart = True

        if isinstance(contract, Bag):
            logger.error("Market depth does not support spreads!")
            return

        self.depthState[contract] = self.ib.reqMktDepth(
            contract, numRows=55, isSmartDepth=useSmart
        )

        t = self.depthState[contract]
        i = 0

        # loop for up to a second until bids or asks are populated
        while not (t.domBids or t.domAsks):
            i += 1
            await asyncio.sleep(0.001)

            if not (t.domBids or t.domAsks):
                logger.warning(
                    "[{}] Depth not populated. Failing warm-up check {}",
                    contract.localSymbol,
                    i,
                )

                if i > 20:
                    logger.error("Depth not populated in expected time?")
                    return

                await asyncio.sleep(0.15)

        decimal_size = self.state.decimals(contract)

        # now we read lists of ticker.domBids and ticker.domAsks for the depths
        # (each having .price, .size, .marketMaker)
        for i in range(0, self.count):
            if not (t.domBids or t.domAsks):
                logger.error(
                    "{} :: {} of {} :: Result Empty...",
                    contract.symbol,
                    i + 1,
                    self.count,
                )

                await asyncio.sleep(1)
                continue

            # Also show grouped by price with sizes summed and markets joined
            # These frames are a bit of a mess:
            # - Create frame
            # - Group frame by price so same prices use one row
            # - Add all sizes for the same price, and concatenate marketMaker cols
            # - 'convert_dtypes()' so any collapsed rows become None instead of NaN
            #   (aggregation sum of int + NaN = float, but we want int, so we use
            #    int + None = int to stop decimals from appearing in the size sums)
            # - re-sort by price based on side
            #   - bids: high to low
            #   - asks: low to high
            # - Re-index the frame by current sorted positions so the concat joins correctly.
            #   - 'drop=True' means don't add a new column with the previous index value

            # condition dataframe reorganization on the input list existing.
            # for some smaller symbols, bids or asks may not get returned
            # by the flaky ibkr depth APIs

            def becomeDepth(side, sortHighToLowPrice: bool):
                if side:
                    df = pd.DataFrame(side)

                    # count how many exchanges are behind the total volume as well
                    # (the IBKR DOM only gives top of book for each exchange at each price level,
                    #  so we can't actually see underlying "market-by-order" here)
                    # This is essentially just len(marketMaker) for each row.
                    df["xchanges"] = df.groupby("price")["price"].transform("size")

                    aggCommon = dict(size="sum", xchanges="last", marketMaker=list)
                    df = (
                        df.groupby("price", as_index=False)
                        .agg(aggCommon)  # type: ignore
                        .convert_dtypes()
                        .sort_values(by=["price"], ascending=sortHighToLowPrice)
                        .reset_index(drop=True)
                    )

                    # format floats as currency strings with proper cent padding.
                    df["price"] = df["price"].apply(
                        lambda x: f"{Decimal(x).normalize():,.{decimal_size}f}"
                    )

                    # generate a synthetic sum row then add commas to the sums after summing.......
                    df.loc["sum", "size"] = df["size"].sum()
                    df["size"] = df["size"].apply(lambda x: f"{round(x, 8):,}")

                    return df

                return pd.DataFrame([dict(size=0)])

            # bids are sorted HIGHEST PRICE to LOWEST OFFER
            fixedBids = becomeDepth(t.domBids, False)

            # asks are sorted LOWEST OFFER to HIGHEST PRICE
            fixedAsks = becomeDepth(t.domAsks, True)

            fmtJoined = {"Bids": fixedBids, "Asks": fixedAsks}

            # Create an order book with high bids and low asks first.
            # Note: due to the aggregations above, the bids and asks
            #       may have different row counts. Extra rows will be
            #       marked as <NA> by pandas (and we can't fill them
            #       as blank because the cols have been coerced to
            #       specific data types via 'convert_dtypes()')
            both = pd.concat(fmtJoined, axis=1)
            both = both.fillna(-1)

            printFrame(
                both,
                f"{contract.symbol} :: {i + 1} of {self.count} :: {contract.localSymbol} Grouped by Price",
            )

            # Note: the 't.domTicks' field is just the "update feed"
            #       which ib_insync merges into domBids/domAsks
            #       automatically, so we don't need to care about
            #       the values inside t.domTicks

            if i < self.count - 1:
                try:
                    await asyncio.sleep(1)
                except:
                    logger.warning("Stopped during sleep!")
                    break

        # logger.info("Actual depth: {} :: {}", pp.pformat(t.domBidsDict), pp.pformat(t.domAsksDict))
        self.ib.cancelMktDepth(contract, isSmartDepth=useSmart)
        del self.depthState[contract]
