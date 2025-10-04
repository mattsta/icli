"""Command: remove, rm

Category: Live Market Quotes
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.cmds.utils import (
    expand_symbols,
)
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["remove", "rm"])
@dataclass
class IOpQuotesRemove(IOp):
    """Remove live quotes from display."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: expand_symbols(x))]

    async def run(self):
        # we 'reverse sort' here to accomodate deleting multiple quote-by-index positions
        # where we always want to delete from HIGHEST INDEX to LOWEST INDEX because when we
        # delete from HIGH to LOW we delete in a "safe order" (if we delete from LOW to HIGH every
        # current delete changes the index of later deletes so unexpected things get removed)
        # Also, we just de-duplicate the symbol requests into a set in case there are duplicate requests
        # (because it would look weird doing "remove :29 :29 :29 :29" just to consume the same position
        #  as it gets removed over and over again?).
        # BUG NOTE: if you mix different digit lengths (like :32 and :302 and :1 and :1005) this sorted() doesn't
        #           work as expected, but most users shouldn't be having more than 100 live quotes anyway.
        #           We could implement a more complex "detect if ':N' syntax then use natural sort' but it's
        #           not important currently. You can see the incorrect sorting behavior using: `remove :{25..300}`
        sym: str | None
        for sym in sorted(set(self.symbols), reverse=True):
            assert sym
            sym = sym.upper()
            sym, contract = await self.state.positionalQuoteRepopulate(sym)

            if contract:
                # logger.info("Removing quote for: {}", contract)
                if not self.ib.cancelMktData(contract):
                    logger.error("Failed to unsubscribe for: {}", contract)

                symkey = lookupKey(contract)
                try:
                    del self.state.quoteState[symkey]

                    logger.info(
                        "[{} :: {}] Removed: {} ({})",
                        sym,
                        contract.conId,
                        nameForContract(contract),
                        symkey,
                    )
                except:
                    # logger.exception("Failed to cancel?")
                    pass

        # re-run all bag compliance math for attaching tickers since something in our quote state changed
        self.state.complyITickersSharedState()

        # also update current client persistent quote snapshot
        await self.runoplive("qsnapshot")
