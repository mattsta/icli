"""Command: chains

Category: Live Market Quotes
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
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
import asyncio

import dateutil.parser
import prettyprinter as pp  # type: ignore
import whenever


@command(names=["chains"])
@dataclass
class IOpOptionChain(IOp):
    """Print option chains for symbol"""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*symbols",
                convert=lambda x: [z.upper() for z in x],
            )
        ]

    async def run(self):
        # Index cache by symbol AND current date because strikes can change
        # every day even for the same expiration if there's high volatility.
        got = {}

        symbol: str | None
        self.symbols: list[str]
        IS_PREVIEW = self.symbols[-1] == "PREVIEW"
        IS_REFRESH = False
        if self.symbols[-1] == "REFRESH":
            IS_REFRESH = True
            # remove "REFRESH" from symbols so we don't try to look it up
            self.symbols = self.symbols[:-1]

        for symbol in self.symbols:
            # if asking for weeklies, need to correct symbol for underlying quote...
            if symbol == "SPXW":
                symbol = "SPX"

            # our own symbol hack for printing more detailed output here
            if symbol == "PREVIEW":
                continue

            contractFound: Contract | None = None
            if symbol.startswith(":"):
                symbol, contractFound = self.state.quoteResolve(symbol)
                assert symbol

            cacheKey = ("chains", symbol)
            # logger.info("Looking up {}", cacheKey)

            # if found in cache, don't lookup again!
            if not IS_REFRESH:
                if (found := self.cache.get(cacheKey)) and all(list(found.items())):
                    # don't print cached chains by default because this pollutes `Fast` output,
                    # but if the last symbol is "preview" then do show it...
                    if IS_PREVIEW:
                        logger.info(
                            "[{}] Already cached: {}",
                            symbol,
                            pp.pformat(found.items()),
                        )
                    got[symbol] = found
                    continue

            # resolve for option chains lookup

            if contractFound:
                contractExact = contractFound
            else:
                contractExact = contractForName(symbol)

            # if we want to request ALL chains, only provide underlying
            # symbol then mock it as "OPT" so the IBKR resolver sees we
            # are requesting option underlying and not just single-stock
            # contract details.
            # NOTE: we DO NOT POPULATE contractId on OPT/FOP lookups or else
            #       it pins the entire lookup to a single ID. We need our
            #       lookups to search more generally without an id.
            if isinstance(contractExact, Stock):
                # hack/edit contract type to be different for OPTION lookup using Stock underlying details
                contractExact.secType = "OPT"
                contractExact.conId = 0
            elif isinstance(contractExact, Future):
                # hack/edit contract type to be different for FUTURES OPTION lookup using Future underlying details
                contractExact.secType = "FOP"
                contractExact.conId = 0
            elif isinstance(contractExact, Index):
                # sure, is fine too
                # we use the different API for index options, so we need an exact contract id
                (contractExact,) = await self.state.qualify(contractExact)
                pass
            else:
                logger.error(
                    "Unknown contract type requested for chain? Got: {}", contractExact
                )

            # note "fetching" here so it doesn't fire if the cache hit passes.
            # (We only want to log the fetching network requests; just cache reading is quiet)
            logger.info("[chains] Fetching for {} using {}", symbol, contractExact)

            # If full option symbol, get all strikes for the date of the symbol
            if isinstance(contractExact, (Option, FuturesOption)):
                contractExact.strike = 0.00
                # if is option already, use exact date of option...
                useDates = set(
                    [dateutil.parser.parse("20" + symbol[-15 : -15 + 6]).date()]
                )  # type: ignore
            else:
                # Actually, even more, we should use the calendar returned from contract details, but, due to IBKR
                # data problems, we change the gateway to only return 2 future days instead of 10+ future days. sigh.
                # Note: the 'tradingDays()' API returns number of market days in the next N CALENDAR DAYS. So if you want
                #       an entire next month of expirations, you need to request "31" trading days ahead.
                # ALSO NOTE: this doesn't work cleanly _across future expiration boundaries_ for generic symbols.
                #            So if you need the next quarter future expiration, you need to use the exact name and not just '/ES + 100 days' etc.
                # TODO: this fetches many redundant dates because not all symbols trade in all days. We should fix the request system.
                useDates = [
                    datetime.date(d.year, d.month, d.day)  # type: ignore
                    for d in self.state.tradingDays(40)
                ]

            # this request takes between 1 second and 60 seconds depending on ???
            # does IBKR rate limit this endpoint? sometimes the same request
            # takes 1-5 seconds, other times it takes 45-65 seconds.
            # Maybe it's rate limited to one "big" call per minute?
            # (UPDATE: yeah, the docs say:
            #  "due to the potentially high amount of data resulting from such
            #   queries this request is subject to pacing. Although a request such
            #   as the above one will be answered immediately, a similar subsequent
            #   one will be kept on hold for one minute.")
            # So this endpoint is a jerk. You may need to rely on an external data
            # provider if you want to gather full chains of multiple underlyings
            # without waiting 1 minute per symbol.
            # Depending on what you ask for with the contract, it can return between
            # one row (one date+strike), dozens of rows (all strikes for a date),
            # or thousands of rows (all strikes at all future dates).

            # complete return value for caller to use if they need it
            got = {}

            # fetch dates by MONTH so IBKR returns all days for all the months requested
            # (so we don't get stuck requesting day-by-day or non-existing days between instruments
            #  trading daily vs weekly vs monthly vs quarterly etc)
            fetchDates = sorted(set([f"{d.year}{d.month:02}" for d in useDates]))
            logger.info("Fetching for dates: {}", fetchDates)

            logger.info("[{}{}] Fetching strikes...", symbol, fetchDates)

            try:
                # Note: we converted this from reqContractDetailsAsync() to use multiple data providers.
                #       If you have a tradier API key in your environment (TRADIER_KEY=key), we use tradier
                #       directly because it fetches chains in less than 100 ms while IBKR sometimes takes minutes
                #       to fetch many chains at once.
                #
                # long timeout here because these get delayed by IBKR "pacing backoff" API limits sometimes.
                #
                # Note: IBKR API docs mention another function called reqSecDefOptParams() but it returns INVALID date+strike details
                #       so its output is completely useless (it returns a list of dates and a list of strikes, but it doesn't combine
                #       strikes-per-date so you have "all strikes for the next 5 years" against "all dates for the next 5 years" which
                #       isn't valid because of things like split-strikes-vs-weekly-vs-opex-vs-LEAP strikes all combined showing for
                #       all dates which isn't applicable at all, so the output from reqSecDefOptParams() is practically useless).
                strikes = await asyncio.wait_for(
                    self.state.fetchContractExpirations(contractExact, fetchDates),
                    timeout=180,
                )

                # only cahce if strikes exist _AND_ all the values exist
                # (because strikes are returned as {key: strikes} so 'strikes' should always exist on its own
                if strikes and all(list(strikes.values())):
                    # TODO: maybe more logically expire strike caches at the next 1615 available
                    # Expire in 10 calendar days at 1700 ET (to prevent these from expiring IN THE MIDDLE OF A LIVE SESSION).
                    now = whenever.ZonedDateTime.now("US/Eastern")
                    expireAt = now.add(days=10, disambiguate="compatible").replace(
                        hour=17, minute=0, disambiguate="compatible"
                    )
                    expireSeconds = (expireAt - now).in_seconds()

                    self.cache.set(cacheKey, strikes, expire=expireSeconds)  # type: ignore

                got[symbol] = strikes
            except:
                logger.exception(
                    "[{}] Error while fetching contract details...",
                    symbol,
                )

            logger.info("Strikes: {}", pp.pformat(got))

        return got
