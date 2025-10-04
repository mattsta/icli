"""Command: details

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Bag,
)
from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.futsexchanges import FUTS_TICK_DETAIL
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio

import prettyprinter as pp  # type: ignore


@command(names=["details"])
@dataclass
class IOpDetails(IOp):
    """Show the IBKR contract market details for a symbol.

    This is useful to check names/industries/trade dates/algos/exchanges/etc.

    Note: this is _NOT_ included as part of `info` output because `details` requires a
          slower server-side data fetch for the larger market details (which can be big and
          introduce pacing violations if run too much at once).
    """

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", desc="Symbols to check for contract details")]

    async def run(self):
        contracts = []

        for sym in self.symbols:
            # yet another in-line hack for :N lookups because we still haven't created a central abstraction yet...
            _name, c = await self.state.positionalQuoteRepopulate(sym)

            # don't allow meta-contracts (or failed lookups) into the detail request queue
            if isinstance(c, Bag) or not c:
                logger.warning(
                    "[{}] Contract not usable for detail request: {}", sym, c
                )
                continue

            contracts.append(c)

        # If any lookups fail above, remove 'None' results before we fetch full contracts.
        contracts = await self.state.qualify(*contracts)

        # TODO: we should actually cache these detail results and have them expire at the end of every
        #       day (the details include day-changing quantities like next N day lookahead trading sessions,
        #       so the details _do_ change over time, but they _do not_ change within a single day).

        # Map of RuleId to RuleValue
        ruleCache: dict[int, ib_async.objects.PriceIncrement] = dict()

        for contract in contracts:
            try:
                (detail,) = await self.ib.reqContractDetailsAsync(contract)

                # IBKR "rules" map one marketRuleId to one exchange by position in each list.
                # So even though detail shows like "26,26,26,26,..." 16 times it's because it matches 16 exchanges.
                # Also see: https://interactivebrokers.github.io/tws-api/minimum_increment.html

                # For cleaner results, we show the inverse of which rule id is serviced by which exchanges
                # because the rules are primarily by instrument type, so almost all exchanges have the same rules.

                # These "marketRuleIds" show the actual security increments like when options
                # trade $0.01 under $3 then $0.05 over $3 versus $0.05 under $3 then $0.10 over $3 versus $0.01 for all, etc.

                # split rule id string into integers
                ridsAll = [int(x) for x in detail.marketRuleIds.split(",")]

                # map exchanges to their matching rule ids
                exchanges = detail.validExchanges.split(",")
                exchangeRuleMapping = dict(zip(exchanges, ridsAll))

                # fetch non-duplicate rule ids
                # TODO: we should cache these lookup results forever. The underlying marketRuleId results will never change,
                #       so running a network call for each detail attempt it wasteful.
                rids = tuple(set(ridsAll))
                rules = await asyncio.gather(
                    *[self.ib.reqMarketRuleAsync(rid) for rid in rids]
                )

                # map rule ids to result value from lookup
                ruleCache.update(dict(zip(rids, rules)))  # type: ignore

                # logger.info("Got rules of:\n{}", pp.pformat(rules))
                # logger.info("Valid Exchanges: {}", detail.validExchanges)

                # logger.info("Exchange rules: {} (via {})", exchangeRuleMapping, exchanges)

                # FORWARD map of xchange -> rule
                # (we conver the rule list to a tuple so it can then be a dict key when we invert this next)
                exchangesWithRules = {
                    xchange: tuple(ruleCache[rid])
                    for xchange, rid in exchangeRuleMapping.items()
                }

                # REVERSE COLLECTIVE MAP of rule -> exchanges
                rulesWithExchanges = defaultdict(list)
                for xch, rule in exchangesWithRules.items():
                    rulesWithExchanges[rule].append(xch)

                # logger.info("Exchanges with rules: {}", exchangesWithRules)
            except:
                logger.error("Contract details not found for: {}", contract)
                continue

            assert detail.contract

            # Only print ticker if we have an active market data feed already subscribed on this client
            logger.info(
                "[{}] Details: {}", detail.contract.localSymbol, pp.pformat(detail)
            )

            try:
                logger.info(
                    "[{}] Extra: {}",
                    detail.contract.localSymbol,
                    pp.pformat(FUTS_TICK_DETAIL[detail.contract.symbol]),
                )
                logger.info(
                    "[{}] Extra: {}",
                    detail.contract.localSymbol,
                    pp.pformat(FUTS_EXCHANGE[detail.contract.symbol]),
                )
            except:
                # we don't care if this fails, it's just nice if we have the data
                pass

            logger.info(
                "[{}] Trading Sessions: {}",
                detail.contract.localSymbol,
                pp.pformat(detail.tradingSessions()),
            )

            logger.info(
                "[{}] Exchange Rule Pairs:\n{}",
                detail.contract.localSymbol,
                pp.pformat(dict(rulesWithExchanges)),
            )
