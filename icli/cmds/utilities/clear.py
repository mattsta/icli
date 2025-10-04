"""Command: clear

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Bag,
    Contract,
    FuturesOption,
    Option,
)
from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import whenever


@command(names=["clear"])
@dataclass
class IOpClearDetails(IOp):
    """Clear toolbar or account status details"""

    what: str = field(init=False)
    extra: str = field(init=False)

    def argmap(self):
        return [
            DArg("what", convert=str.lower),
            DArg("extra", convert=str.lower, default=""),
        ]

    async def run(self):
        """Remove PnL fields so they will be properly re-populated during the next PnL event."""
        match self.what:
            case "highlow":
                # clear local high/low stats for spreads so they begin re-calculating...
                logger.info("Clearing all High/Low values for spreads...")
                for _symbol, ticker in self.state.quotesPositional:
                    # Only remove options with no bids because otherwise things like vix get deleted.
                    if isinstance(ticker.contract, Bag):
                        ticker.ticker.high = None  # type: ignore
                        ticker.ticker.low = None  # type: ignore
            case "pnl":
                logger.info("Clearing PnL account status fields...")
                for k in sorted(self.state.accountStatus.keys()):
                    if "pnl" in k.lower():
                        del self.state.accountStatus[k]
            case "noquote":
                logger.info("Clearing quotes without bids...")
                removals = []

                for idx, (_symbol, ticker) in enumerate(self.state.quotesPositional):
                    # Only remove options with no bids because otherwise things like vix get deleted.
                    if isinstance(ticker.contract, (Option, Bag, FuturesOption)):
                        if ticker.bid is None or ticker.ask is None:
                            removals.append(f":{idx}")

                if not removals:
                    logger.info("No quotes found to remove!")
                    return

                removal = " ".join(removals)
                logger.info("Found for removal: {}", removal)
                await self.runoplive("rm", removal)
            case "opt" | "options":
                # Remove ALL option quotes

                if self.extra:
                    logger.warning(
                        "[clear {}] No arguments needed here. Did you mean `clear expire {}` instead?",
                        self.what,
                        self.extra,
                    )
                    return

                logger.info("Removing ALL option quotes...")
                removals = []

                # Why are we removing only by position? We can't reference spreads/bags by any name
                # other than their actual position keys, so just position key everything. The positions
                # are stable as long as nothing else adds keys between the removals. We could technically
                # add an extra "add/remove toolbar quotes lock" somewhere, but it's not necessary currently.
                for idx, (_symbol, ticker) in enumerate(self.state.quotesPositional):
                    if isinstance(ticker.contract, (Option, FuturesOption, Bag)):
                        removals.append(f":{idx}")

                if not removals:
                    logger.info("No quotes found to remove!")
                    return

                removal = " ".join(removals)
                logger.info("Found for removal: {}", removal)
                await self.runoplive("rm", removal)
            case "exp" | "expired":
                # Remove every expired option either older than today or today if it's past the 0DTE closing time
                now = whenever.ZonedDateTime.now("US/Eastern")

                # one extra tick: if *tomorrow* is requested, we increase our "next day" math by one!
                extraDayBooster = 1 if self.extra == "tomorrow" else 0
                self.extra = "today"

                # By default we delete quotes OLDER than today, but you can optionally request deleting live TODAY quotes too.
                # if we are after the close of 0DTE trading, also remove TODAY's quotes too
                # (by pretending today is tomorrow so tomorrow's today deletion would delete today too)
                if (now.hour, now.minute) >= (16, 0) or self.extra == "today":
                    now = now.add(days=1 + extraDayBooster, disambiguate="compatible")

                today = f"{now.year:04}{now.month:02}{now.day:02}"

                removals = []
                logger.info("Removing EXPIRED option quotes OLDER than {}...", today)
                for idx, (_symbol, ticker) in enumerate(self.state.quotesPositional):
                    contract = ticker.contract
                    if isinstance(contract, (Option, FuturesOption)):
                        if contract.lastTradeDateOrContractMonth < today:
                            removals.append(f":{idx}")
                    elif isinstance(contract, Bag):
                        # if it's a bag, we look at EACH LEG to see if ANY leg is older than today
                        innerContracts = await self.state.qualify(
                            *[Contract(conId=x.conId) for x in contract.comboLegs]
                        )

                        if any(
                            [
                                x.lastTradeDateOrContractMonth < today
                                for x in innerContracts
                                if x.lastTradeDateOrContractMonth
                            ]
                        ):
                            removals.append(f":{idx}")

                if not removals:
                    logger.info("No quotes found to remove!")
                    return

                removal = " ".join(removals)
                logger.info("Found for removal: {}", removal)
                await self.runoplive("rm", removal)

            case "unused":
                logger.info(
                    "[{}] Removing option quotes not used by spreads...", self.what
                )
                # Remove single option quotes not used by listed spreads
                allSingleLegIds = set()
                allBagLegIds = set()
                save = set()

                # here, instead of removing by position, we can remove by contract ID since our quote add/removal
                # system parses pure numerical symbol input as IBKR contract IDs (and here we are not removing Bag
                # quote rows, so all rows we remove can be addressed with the single contractId on their own).
                for idx, (_symbol, ticker) in enumerate(self.state.quotesPositional):
                    contract = ticker.contract
                    if isinstance(contract, (Option, FuturesOption)):
                        allSingleLegIds.add(contract.conId)
                    elif isinstance(contract, Bag):
                        allBagLegIds |= set([x.conId for x in contract.comboLegs])
                    else:
                        # else, if it's something else, add it to a special "DO NOT DELETE" collection
                        save.add(contract.conId)

                # Remove all contracts NOT PRESENT in the bag leg ids
                setremove = allSingleLegIds ^ allBagLegIds

                # Re-add anything not an option in case we had multi-instrument bags
                setremove -= save

                if setremove:
                    removal = " ".join(map(str, setremove))
                    logger.info(
                        "Found single contracts not used by spreads: {}", removal
                    )
                    await self.runoplive("rm", removal)
