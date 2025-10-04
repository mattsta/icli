"""Command: report

Category: Portfolio
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Contract,
)
from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import numpy as np
import pandas as pd
import prettyprinter as pp  # type: ignore


@command(names=["report"])
@dataclass
class IOpOrderReport(IOp):
    """Show position details using the OrderMgr logged trade records."""

    stopPct: float = field(init=False)

    def argmap(self):
        # TODO: filter for symbols
        return [
            DArg(
                "stopPct",
                convert=float,
                default=0.10,
                desc="Use as stop calculation percentage. 10% == 0.10 for this input",
            )
        ]

    async def run(self):
        # position_groups compares every trade for every live symbol
        # to find positions having shared order execution for tracking
        # down spreads.
        groups = self.state.ordermgr.position_groups()

        if not groups:
            logger.info("No saved positions to report!")
            return None

        # 'groups' is a dict of {conId: PositionGroup(positions=set[Position])}

        # Steps:
        #  1. resolve each conId into a full contract so we can report on its details
        #  2. for each top-level conId, print its PositionGroup membership (more conIds) and summary data.

        try:
            contracts: Sequence[Contract] = await self.state.qualify(
                *[Contract(conId=x) for x in groups.keys()]  # type: ignore
            )
        except:
            logger.error(
                "Sorry, your contracts are expired can't can be resolved. You may need to remove expired positions:\n{}",
                pp.pformat(groups),
            )
            return None

        digits = self.state.decimals(contracts[0])

        conIdMap: dict[Hashable, Contract] = {
            contract.conId: contract for contract in contracts
        }

        # save already-shown contract ids so we don't repeat them
        topLevelPresented: set[int] = set()

        summaries = {}

        now = self.state.now

        def livePriceFetcher(key):
            # for our positions, keys are contractIds, so we can look up quotes by id (if they exist).
            # (yes, this is redundant, we should probably re-work the entire quote system to use contractId instead
            #  of string names, but we originally made string names so we could easily avoid adding duplicate quote
            #  requests at time of the request (e.g. "add AAPL" can easily reject AAPL without looking it up again))
            if q := self.state.quoteState.get(
                self.state.contractIdsToQuoteKeysMappings.get(key)  # type: ignore
            ):
                if q.bid is not None and q.ask is not None:
                    return (q.bid + q.ask) / 2

            # else, either not found _or_ no bid/ask, so we can't run a profit calc or adjust stops currently.
            return np.nan

        for conId, group in groups.items():
            # if we already presented a contract as an element of another group, don't print
            # it as another top-level output
            if conId in topLevelPresented:
                continue

            components = []
            details = []
            for position in group.positions:
                # fetch contract details from id
                c = conIdMap[position.key]

                # generate readable name (the group only has numeric contract ID, so we need to show something useful)
                components.append(
                    f"[{c.secType} {c.localSymbol}] [qty {position.qty}] [avg ${position.average_price:,.{digits}f}]"
                )
                assert isinstance(position.key, int)

                topLevelPresented.add(position.key)

                for trade in position.trades.values():
                    details.append(
                        "\t[key {}] [{}] [ord {}] {:,} @ ${:,.{}f} ({} ago)".format(
                            group.key,
                            c.localSymbol,
                            trade.orderid,
                            trade.qty,
                            trade.average_price,
                            digits,
                            (now - trade.timestamp).in_words(),  # type: ignore
                        )
                    )

            name = " :: ".join(components)

            summary = group.summary(stopPct=self.stopPct, priceFetcher=livePriceFetcher)
            summaries[group.key] = summary

            logger.info("[key {}] [{}]", group.key, name)
            logger.info(
                "[key {}] [{}] :: TRADES\n{}", group.key, name, "\n".join(details)
            )

            logger.info("[key {}] OPEN: {}", group.key, group.open("LIM"))
            logger.info("[key {}] CLOSE: {}", group.key, group.close("LIM"))
            logger.info(
                "[key {}] ENTR ({:>6,.2f} % max incr): {}",
                group.key,
                self.stopPct * 100,
                group.start(stopPct=self.stopPct, algo="LIM"),
            )
            logger.info(
                "[key {}] EXIT ({:>6,.2f} % max loss): {}",
                group.key,
                self.stopPct * 100,
                group.stop(stopPct=self.stopPct, algo="LIM"),
            )
            logger.info(
                "[key {}] EXIT ({:>6,.2f} % max wins): {}",
                group.key,
                -self.stopPct * 100,
                group.stop(stopPct=-self.stopPct, algo="LIM"),
            )
            logger.info("---")

        df = pd.DataFrame.from_dict(summaries, orient="index")
        printFrame(df.convert_dtypes(), "Groups Summary")
