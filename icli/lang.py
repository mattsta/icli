import bisect
import calendar
import copy

import fnmatch
import itertools
import math
import os
import pathlib
import random
import statistics
import time
import functools
import sys

from dataclasses import dataclass, field

import pytz
from typing import *

from collections import defaultdict
from decimal import Decimal

import mutil.dispatch
import mutil.expand
import numpy as np

import pandas as pd

from ib_async import (
    Bag,
    CommissionReport,
    Contract,
    Execution,
    Fill,
    Order,
    OrderStatus,
    PortfolioItem,
    # For simulation testing
    Trade,
    TradeLogEntry,
)
from loguru import logger
from mutil.bgtask import BGSchedule
from mutil.dispatch import DArg
from mutil.frame import printFrame
from mutil.numeric import fmtPrice, roundnear
from icli.helpers import *
import asyncio

import aiohttp
import dateutil.parser
import whenever

import prettyprinter as pp  # type: ignore
from questionary import Choice
from tradeapis import ifthen as ifthen
from tradeapis import ifthen_dsl as ifthen_dsl
from tradeapis.orderlang import (
    Calculation,
    DecimalLongShares,
    DecimalPrice,
    DecimalShortShares,
)

from .futsexchanges import FUTS_TICK_DETAIL

pp.install_extras(["dataclasses"], warn_on_error=False)

# TODO: convert to proper type and find all misplaced uses of "str" where we want Symbol.
# TODO: also break out Symbol vs LocalSymbol usage
type Symbol = str
type Price = Decimal

if TYPE_CHECKING:
    # import icli under TYPE_CHECKING guard because it's actually a circular import and we can't
    # do this at runtime (which is why we reference icli.cli as 'self.state' everywhere), but
    # mypy/typing has more flexibility with resolving circular imports to allow type checks to
    # see objects throughout a project easier.
    import icli.cli as typecheckingicli

# The choices map nice user strings to the lookup map in orders.order(orderType)
# for dynamically returning an order instance based on string name...
ORDER_TYPE_Q = Q(
    "Order Type",
    choices=[
        Choice("Limit", "LMT"),
        Choice("Adaptive Fast", "LMT + ADAPTIVE + FAST"),
        Choice("Adaptive Slow", "LMT + ADAPTIVE + SLOW"),
        Choice("Peg Primary (RELATIVE)", "REL"),
        Choice("MidPrice", "MIDPRICE"),
        Choice("Market", "MKT"),
        Choice("Stop", "STP"),
        Choice("Limit If Touched", "LIT"),
        Choice("Market If Touched", "MIT"),
        Choice("Adaptive Fast Market", "MKT + ADAPTIVE + FAST"),
        Choice("Adaptive Slow Market", "MKT + ADAPTIVE + SLOW"),
        Choice("Market on Open (MOO)", "MOO"),
        Choice("Market on Close (MOC)", "MOC"),
        Choice("Market to Limit", "MTL"),
        Choice("Market with Protection (Futures)", "MKT PRT"),
        Choice("Stop with Protection (Futures)", "STOP PRT"),
        Choice("Peg to Midpoint (IBKR Dark Pool Routing)", "PEG MID"),
    ],
)


def addRowSafe(df, name, val):
    """Weird helper to stop a pandas warning.

    Fixes this warning pandas apparently is now making for no reason:
    FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
                   In a future version, this will no longer exclude empty or all-NA columns when determining
                   the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
    """
    return pd.concat([df, pd.Series(val, name=name).to_frame().T], ignore_index=False)


def automaticLimitBuffer(contract, isBuying: bool, price: float) -> float:
    """Given a contract and a target price we want to meet, create a limit order bound so we don't slide out of the quote spread.

    This is basically an extra effort way of just doing IBKR's built-in ADAPTIVE FAST algo? Maybe? but we trust it more
    because we do it ourself? Also see potentially (depending on instrument): Market-to-Limit or just the Adaptive Market algos?
    """
    EQUITY_BOUNDS = 1.0025
    OPTION_BOUNDS = 1.15
    OPTIONS_BOOST = 1.333

    if isBuying:
        # if position IS BUYING, we want to chase HIGHER PRICES TO GET THE ORDER
        limit = price * EQUITY_BOUNDS

        if isinstance(contract, (Option, FuturesOption)):
            # options have deeper exit floor criteria because their ranges can be wider.
            # the IBKR algo is "adaptive fast" so it should *try* to pick a good value in
            # the spread without immediately going to market, but ymmv.
            limit = price * OPTION_BOUNDS

            # if price is too small, it may be moving faster, so increase the limit slightly more
            # (this is still always bound by the market spread anyway; we're basically doing an
            #  excessively high effort market order but just trying to protect against catastrophic fills)
            if limit < 2:
                limit = limit * OPTIONS_BOOST
    else:
        # else, position IS SELLING, we want to chase LOWER PRICES to GET THE ORDER
        limit = price / EQUITY_BOUNDS

        if isinstance(contract, (Option, FuturesOption)):
            # options have deeper exit floor criteria because their ranges can be wider.
            # the IBKR algo is "adaptive fast" so it should *try* to pick a good value in
            # the spread without immediately going to market, but ymmv.
            limit = price / OPTION_BOUNDS

            # (see logic/rationale above)
            if limit < 2:
                limit = limit / OPTIONS_BOOST

    return limit


def expand_symbols(symbols):
    # pre-process input strings so we can use symbols like SPXW231103{P,C}04345000
    useSymbols = set()
    for sym in set(symbols):
        useSymbols |= set(mutil.expand.expand_string_curly_braces(sym))

    return useSymbols


@dataclass
class IOp(mutil.dispatch.Op):
    """Common base class for all operations.

    Just lets us have a single point for common settings across all ops."""

    # Note: this is a quoted annotation so python ignores but mypy can still use it
    state: "typecheckingicli.IBKRCmdlineApp"

    def __post_init__(self):
        # for ease of use, populate state IB into our own instance
        assert self.state
        self.ib = self.state.ib
        self.cache = self.state.cache

    def runoplive(self, cmd, args=""):
        # wrapper for things like:
        #        strikes = await self.state.dispatch.runop(
        #            "chains", self.symbol, self.state.opstate
        #        )
        return self.state.dispatch.runop(cmd, args, self.state.opstate)

    def task_create(self, *args, **kwargs):
        return self.state.task_create(*args, **kwargs)


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


@dataclass
class IOpSetEnvironment(IOp):
    """Read or Write a global environment setting for this current client session.

    For a list of settable options, just run `set show`.
    To view the current value of an option, run `set [key]` with no value.
    To delete a key, use an empty value for the argument as `set [key] ""`.
    """

    key: str = field(init=False)
    val: list[str] = field(init=False)

    def argmap(self):
        return [DArg("key", default=""), DArg("*val", default="")]

    async def run(self):
        if not (self.key or self.val):
            # if no input, just print current state
            self.state.updateGlobalStateVariable("", None)
            return

        val = self.val[0] if self.val else None

        self.state.updateGlobalStateVariable(self.key, val)


@dataclass
class IOpUnSetEnvironment(IOp):
    """Remove an environment variable (if set)."""

    key: str = field(init=False)

    def argmap(self):
        return [DArg("key", default="")]

    async def run(self):
        if not self.key:
            # if no input, just print current state
            self.state.updateGlobalStateVariable("", None)
            return

        self.state.updateGlobalStateVariable(self.key, "")


@dataclass
class IOpSay(IOp):
    """Speak a custom phrase provided as arguments.

    Can be used as a standalone command or combined with scheduled events to create
    speakable events on a delay."""

    what: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*what")]

    async def run(self):
        content = " ".join(self.what)
        self.task_create(content, self.state.speak.say(say=content))


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


@dataclass
class IOpCash(IOp):
    def argmap(self):
        return []

    async def run(self):
        result = {
            "Avail Full": [
                self.state.accountStatus["AvailableFunds"],
                self.state.accountStatus["AvailableFunds"] * 2,
                self.state.accountStatus["AvailableFunds"] * 4,
            ],
            "Avail Buffer": [
                self.state.accountStatus["AvailableFunds"] / 1.10,
                self.state.accountStatus["AvailableFunds"] * 2 / 1.10,
                self.state.accountStatus["AvailableFunds"] * 4 / 1.10,
            ],
            "Net Full": [
                self.state.accountStatus["NetLiquidation"],
                self.state.accountStatus["NetLiquidation"] * 2,
                self.state.accountStatus["NetLiquidation"] * 4,
            ],
            "Net Buffer": [
                self.state.accountStatus["NetLiquidation"] / 1.10,
                self.state.accountStatus["NetLiquidation"] * 2 / 1.10,
                self.state.accountStatus["NetLiquidation"] * 4 / 1.10,
            ],
        }

        printFrame(
            pd.DataFrame.from_dict(
                {k: [f"${v:,.2f}" for v in vv] for k, vv in result.items()},
                orient="index",
                columns=["Cash", "Overnight", "Day"],
            )
        )


@dataclass
class IOpCalendar(IOp):
    """Just show a calendar!"""

    year: list[str] | None = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*year",
                desc="Year for your calendar to show (if not provided, just use current year)",
            )
        ]

    async def run(self):
        try:
            assert self.year
            year = int(self.year[0])
        except:
            year = whenever.ZonedDateTime.now("US/Eastern").year

        # MURICA
        # (also lol for this outdated python API where you have to globally set the calendar start
        #  date for your entire environment!)
        calendar.setfirstweekday(calendar.SUNDAY)
        logger.info("[{}] Calendar:\n{}", year, calendar.calendar(year, 1, 1, 6, 3))


@dataclass
class IOpCalculator(IOp):
    """Just show a calculator!"""

    parts: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*parts", desc="Calculator input")]

    async def run(self):
        cmd = " ".join(self.parts)

        try:
            logger.info("[{}]: {:,.4f}", cmd, self.state.calc.calc(cmd))
        except Exception as e:
            logger.warning("[{}]: calculation error: {}!", cmd, e)


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


@dataclass
class IOpInfo(IOp):
    """Show the underlying IBKR contract object for a symbol.

    This is mainly useful to verify the IBKR details or extract underlying contract IDs
    for other debugging or one-off usage purposes."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: sorted(expand_symbols(x)))]

    async def run(self):
        contracts = []

        for sym in self.symbols:
            _name, contract = await self.state.positionalQuoteRepopulate(sym)
            contracts.append(contract)

        # remove anything not actually available (typos mainly like trying to look up :!3 which is None)
        contracts = list(filter(None, contracts))

        # Print contracts first because not all contracts qualify for the extra metadata population,
        # but we still want to print something the user requested at least.
        logger.info("Inbound Contracts:\n{}", pp.pformat(contracts))

        # If any lookups fail above, remove 'None' results before we fetch full contracts.
        qcontracts = await self.state.qualify(*contracts)
        logger.info("Qualified Contracts:\n{}", pp.pformat(qcontracts))

        for contract in qcontracts:
            # if contract didn't qualify, we can't do anything more than just print it as we did above
            # (though, bags *never* have a top-level contract id, so we let them pass through)
            if not (isinstance(contract, Bag) or contract.conId):
                continue

            digits = self.state.decimals(contract)

            # Only print ticker if we have an active market data feed already subscribed on this client
            symkey = lookupKey(contract)
            if iticker := self.state.quoteState.get(symkey, None):
                ticker = iticker.ticker

                # The 'pprint' module doesn't use our nice __repr__ override which removes all nan/None fileds (sometimes dozens per ticker),
                # so let's hack around it by printing the formatted dataclass, splitting by comma lines,
                # removing rows with nan values, then just re-assembling it.

                # drop created so it doesn't print
                ticker.created = None  # type: ignore

                assert ticker.contract

                prettyTicker = ",".join(
                    filter(lambda x: "=None" not in x, pp.pformat(ticker).split(","))
                )
                logger.info("Ticker:\n{}", prettyTicker)

                if ticker.histVolatility:
                    logger.info(
                        "[{}] Historical Volatility: {:,.{}f}",
                        ticker.contract.localSymbol,
                        ticker.histVolatility,
                        digits,
                    )

                if ticker.impliedVolatility:
                    logger.info(
                        "[{}] Implied Volatility: {:,.{}f}",
                        ticker.contract.localSymbol,
                        ticker.impliedVolatility,
                        digits,
                    )

                if ticker.histVolatility and ticker.impliedVolatility:
                    if ticker.histVolatility < ticker.impliedVolatility:
                        logger.info(
                            "[{}] Volatility: RISING ({:,.{}f} %)",
                            ticker.contract.localSymbol,
                            100
                            * ((ticker.impliedVolatility / ticker.histVolatility) - 1),
                            digits,
                        )
                    else:
                        logger.info(
                            "[{}] Volatility: FALLING ({:,.{}f} %)",
                            ticker.contract.localSymbol,
                            100
                            * ((ticker.histVolatility / ticker.impliedVolatility) - 1),
                            digits,
                        )

                if ticker.last:
                    logger.info(
                        "[{}] Last: ${:,.{}f} x {}",
                        ticker.contract.localSymbol,
                        ticker.last,
                        digits,
                        int(ticker.lastSize)
                        if int(ticker.lastSize) == ticker.lastSize
                        else ticker.lastSize,
                    )

                # if this is a bag of things, print each underlying symbol too...
                if isinstance(ticker.contract, Bag):
                    logger.info("Bag has {} legs:", len(ticker.contract.comboLegs))
                    legs = zip(
                        ticker.contract.comboLegs,
                        await self.state.qualify(
                            *[
                                Contract(conId=x.conId)
                                for x in ticker.contract.comboLegs
                            ]
                        ),
                    )
                    for legSrc, legContract in legs:
                        logger.info(
                            "    {:>4} {:>3} {:<} ({:<})",
                            legSrc.action,
                            legSrc.ratio,
                            legContract.localSymbol,
                            legContract.localSymbol.replace(" ", ""),
                        )

                    # also provide an easy quote add syntax for moving this around if we want to
                    logger.info(
                        "{}",
                        " ".join(
                            [
                                f"{leg.action} {leg.ratio} {leg.conId}"
                                for leg in ticker.contract.comboLegs
                            ]
                        ),
                    )

                def tickTickBoom(current, prev, name, xchanges=None, xsize=None):
                    # don't print anything if our data is invalid.
                    # invalid can be: NaNs, after hours prices of -1 or 0, etc.
                    if not (current and prev):
                        return

                    udl = "FLAT"
                    amt = current - prev
                    if amt > 0:
                        udl = "UP"
                    elif amt < 0:
                        udl = "DOWN"

                    xchangeDetails = ""

                    if xchanges:
                        sz = int(xsize) if int(xsize) == xsize else xsize
                        xchangeDetails = f" @ ${current:,.{digits}f} x {sz:,} on {len(xchanges)} exchanges"

                    assert ticker.contract
                    logger.info(
                        "[{}] {} tick {} (${:,.{}f}){}",
                        ticker.contract.localSymbol,
                        name,
                        udl,
                        amt,
                        digits,
                        xchangeDetails,
                    )

                tickTickBoom(
                    ticker.bid,
                    ticker.prevBid,
                    "BID",
                    ticker.bidExchange,
                    ticker.bidSize,
                )
                tickTickBoom(
                    ticker.ask,
                    ticker.prevAsk,
                    "ASK",
                    ticker.askExchange,
                    ticker.askSize,
                )
                tickTickBoom(ticker.last, ticker.prevLast, "LAST")

                # protect against ask being -1 or NaN thanks to weird IBKR data issues when markets aren't live
                if ticker.bid and ticker.ask:
                    logger.info(
                        "[{}] Spread: ${:,.{}f} (± ${:,.{}f})",
                        ticker.contract.localSymbol,
                        ticker.ask - ticker.bid,
                        digits,
                        (ticker.ask - ticker.bid) / 2,
                        digits,
                    )

                if ticker.halted:
                    logger.warning("[{}] IS HALTED!", ticker.contract.localSymbol)

                trf = iticker.emaTradeRate.logScoreFrame(0)
                printFrame(
                    trf,
                    f"Trade Rate Stats [scores [prev {iticker.emaTradeRate.diffPrevLogScore:.10f}] [vwap {iticker.emaTradeRate.diffVWAPLogScore:.10f}]]",
                )

                tvf = iticker.emaVolumeRate.logScoreFrame(2)
                printFrame(
                    tvf,
                    f"Volume Rate Stats [scores [prev {iticker.emaVolumeRate.diffPrevLogScore:.10f}] [vwap {iticker.emaVolumeRate.diffVWAPLogScore:.10f}]]",
                )

                if isinstance(iticker.contract, (Option, FuturesOption, Bag)):
                    vfi = iticker.emaIV.logScoreFrame(3)
                    printFrame(
                        vfi,
                        f"IV Stats [scores [prev {iticker.emaIV.diffPrevLogScore:.10f}] [vwap {iticker.emaIV.diffVWAPLogScore:.10f}]]",
                    )

                    dfi = iticker.emaDelta.logScoreFrame(3)
                    printFrame(
                        dfi,
                        f"Delta Stats [scores [prev {iticker.emaDelta.diffPrevLogScore:.10f}] [vwap {iticker.emaDelta.diffVWAPLogScore:.10f}]]",
                    )

                fi = iticker.ema.logScoreFrame(digits)
                printFrame(
                    fi,
                    f"Price Stats [scores [prev {iticker.ema.diffPrevLogScore:.22f}] [vwap {iticker.ema.diffVWAPLogScore:.22f}]]",
                )

                # TODO: have .anaylize return a dataframe too i guess
                qfresults = iticker.quoteflow.analyze()

                logger.info(
                    "[{}] QuoteFlow Duration: {:,.3f} s",
                    ticker.contract.localSymbol,
                    qfresults["duration"],
                )

                logger.info(
                    "[{}] QuoteFlow UP Average Time: {}",
                    ticker.contract.localSymbol,
                    pp.pformat(
                        {
                            f"${k}": f"{v:,.3f} s ({qfresults['uplen'][k]})"
                            for k, v in qfresults["upspeed"].items()
                        }
                    ),
                )
                logger.info(
                    "[{}] QuoteFlow DOWN Average Time: {}",
                    ticker.contract.localSymbol,
                    pp.pformat(
                        {
                            f"${k}": f"{v:,.3f} s ({qfresults['downlen'][k]})"
                            for k, v in qfresults["downspeed"].items()
                        }
                    ),
                )

                for lookback, atr in iticker.atrs.items():
                    logger.info(
                        "[{}] ATR [{}]: {:,.{}f}",
                        ticker.contract.localSymbol,
                        lookback,
                        atr.current,
                        digits,
                    )

                try:
                    # 'statistics' throws an error if there's not enough history yet
                    qs = statistics.quantiles(iticker.history, n=7, method="inclusive")

                    bpos = bisect.bisect_left(qs, iticker.ema[0])
                    qss = [f"{x:,.{digits}f}" for x in qs]
                    qss.insert(bpos, "[X]")

                    low = min(iticker.history)
                    high = max(iticker.history)

                    logger.info(
                        "[{}] stats (from {}): [range {:,.{}f}] [min {:,.{}f}] [max {:,.{}f}] [std {:,.{}f}]",
                        ticker.contract.localSymbol,
                        len(iticker.history),
                        high - low,
                        digits,
                        low,
                        digits,
                        high,
                        digits,
                        statistics.stdev(iticker.history),
                        digits,
                    )

                    logger.info(
                        "[{}] range: {}",
                        ticker.contract.localSymbol,
                        " :: ".join(qss),
                    )
                except:
                    # logger.exception("what?")
                    pass

                if iticker.bags:
                    logger.info(
                        "[{}] Bags ({}): {}",
                        ticker.contract.localSymbol,
                        len(iticker.bags),
                        [x.contract.comboLegs for x in iticker.bags],
                    )

                if iticker.legs:
                    logger.info(
                        "[{}] Tracking Legs ({}):",
                        ticker.contract.localSymbol,
                        len(iticker.legs),
                    )

                    for ratio, leg in iticker.legs:
                        if leg:
                            logger.info("    [ratio {:>2}] {}", ratio, leg.contract)
                        else:
                            logger.warning("    LEG NOT PRESENT")

                # these are filtered to remove None models when they are being populated during startup...
                greeks = {
                    k: v
                    for k, v in {
                        "bid": ticker.bidGreeks,
                        "ask": ticker.askGreeks,
                        "last": ticker.lastGreeks,
                        "model": ticker.modelGreeks,
                    }.items()
                    if v
                }

                df = pd.DataFrame.from_dict(greeks, orient="index")

                # only print our fancy table if it actually exists and if this holds potentially time or volatility risk premium
                if not df.empty and isinstance(
                    ticker.contract, (Bag, Option, FuturesOption)
                ):
                    # remove rows with broken theta values showing up like '-0.000000'
                    # actually, don't do this because we have postive theta on reported spreads and this just drops them all.
                    # df = df[df.theta < -0.0001]
                    # if df.empty:
                    #     continue

                    # make a column for what percentage of theta is the current option price
                    # (basically: your daily rollover loss percentage if the price doesn't move overnight)
                    df["theta%"] = round(df.theta / df.optPrice, 2)
                    df["delta%"] = round(df.delta / df.optPrice, 2)

                    # theta/delta tells you how much the underlying must go up the next day to compensate
                    # for the theta decay.
                    # e.g. delta 0.10 and theta -0.05 means the underlying must go up $0.50 the next day to remain flat.
                    #      delta 0.03 and theta -0.14 means the underlying must go up $5 the next day to remain flat.
                    # technically we should be using charm here (delta for the 'next day'), but this is close enough for now.
                    df["Θ/Δ"] = round(-df.theta / df.delta, 2)

                    # provide rough (ROUGH) estimates for 3 days into the future accounting for theta
                    # (note: theta is negative, so we just add it. also note: theta doesn't decay linearly,
                    #        so these calculations are not _exact_, but it serves as a mental checkpoint to compare against).
                    df["day+1"] = round(df.optPrice + df.theta, 2).clip(lower=0)
                    df["day+2"] = round(df.optPrice + df.theta * 2, 2).clip(lower=0)
                    df["day+3"] = round(df.optPrice + df.theta * 3, 2).clip(lower=0)

                    # remove always empty columns
                    del df["pvDividend"]
                    del df["tickAttrib"]

                    printFrame(df, "Greeks Table")

                    # only show summary greeks if we have more than one greeks row to compare against
                    if len(df) > 1:
                        min_max_mean = df.agg(["min", "max", "mean"])
                        printFrame(min_max_mean, "Summary Greeks Table")


@dataclass
class IOpReconnect(IOp):
    """Run a full gateway reconnect cycle (primarily for debugging)"""

    shutdown: bool = field(init=False)

    def argmap(self):
        return [
            DArg(
                "shutdown",
                default=False,
                convert=lambda x: x.lower()
                in {"stop", "true", "shutdown", "goodbye", "bye"},
                desc="Whether to disconnect but NOT reconnect",
            )
        ]

    async def run(self):
        # Use the event framework to trigger the reconnect event handler.

        # if true, we disconnect then DO NOT RECONNECT
        # (note: we have no "connect" command, so this will leave your client in a
        #        'disconnected idle' state until you fully exit and restart)
        if self.shutdown:
            self.state.exiting = True

        got = self.state.ib.disconnect()
        logger.warning("{}", got)


@dataclass
class IOpAlert(IOp):
    """Configure in-icli alert settings for crossovers and level breaches."""

    symbol: str = field(init=False)
    builder: str = field(init=False)
    data: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg(
                "setup",
                desc="How to configure 'data' field for symbol",
                convert=str.lower,
            ),
            DArg(
                "*data",
                desc="Which field of data to edit alert settings for",
                convert=lambda x: list(map(str.lower, x)),
            ),
        ]

    async def run(self):
        foundSymbol, contract = await self.state.positionalQuoteRepopulate(self.symbol)

        if not foundSymbol:
            logger.error("Symbol not found? Can't update alert state!")
            return

        self.symbol = foundSymbol
        assert contract

        logger.info("[{}] Updating alert settings...", self.symbol)

        disable = self.builder in {"off", "no", "false", "0", "disable"}
        enable = not disable

        show = self.builder == "show"

        # one one big OFF message for ALL symbols
        if (not enable) and not self.data:
            ...

        symkey = lookupKey(contract)
        iticker = self.state.quoteState[symkey]

        # else, we have at least one data field to consider
        match self.data[0]:
            case "bar":
                # minor hack to just print the current full alert level state
                if show:
                    logger.info("{}", pp.pformat(iticker.levels))
                    return

                # if sub-fields, set them as populated
                # (this references bar size in seconds, so 86400 == 1 day bar, etc)
                if len(self.data) > 1:
                    for level in self.data[1:]:
                        if found := iticker.levels.get(int(level)):
                            found.enabled = enable
                            logger.info("Now enabled={} for {}", enable, found)
                else:
                    for l in iticker.levels.values():
                        l.enabled = enable
                        logger.info("Now enabled={} for {}", enable, l)


@dataclass
class IOpDayDumper(IOp):
    """Save bar history for a symbol to disk using IBKR data APIs."""

    symbol: str = field(init=False)
    back: int = field(init=False)
    interval: str = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg("back", convert=int, default=7),
            DArg("interval", default="1 day"),
        ]

    async def run(self):
        foundSymbol, originalContract = await self.state.positionalQuoteRepopulate(
            self.symbol
        )

        if not foundSymbol:
            logger.error("Symbol not found? Can't perform a lookup!")
            return

        self.symbol = foundSymbol
        assert originalContract

        # For futures, use continuous full historical representation instead of a local expiration date
        # (also, do this on a COPY because we don't want to overwite the shared/cached contract object)
        contract: Contract
        if isinstance(originalContract, Future):
            contract = copy.copy(originalContract)
            contract.secType = "CONTFUT"
        else:
            contract = originalContract

        # fetch data
        found = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            # IBKR only accepts 'D' requests for <= 365 days, else it requires full years of retrieval
            durationStr=f"{self.back} D"
            if self.back <= 365
            else f"{math.ceil(self.back / 365)} Y",
            # Valid bar sizes are:
            # 1 secs	5 secs	10 secs	15 secs	30 secs 1 min	2 mins	3 mins	5 mins	10 mins	15 mins	20 mins	30 mins 1 hour	2 hours	3 hours	4 hours	8 hours 1 day 1 week 1 month
            barSizeSetting=self.interval,
            whatToShow="TRADES",
            useRTH=False,
            keepUpToDate=False,
            timeout=6,
        )

        # augment data a little

        def wwma(values, n):
            return values.ewm(
                alpha=1 / n, min_periods=n, ignore_na=True, adjust=False
            ).mean()

        def atr(df, n=14):
            data = df.copy()
            high = data.high
            low = data.low
            close = data.close
            data["tr0"] = abs(high - low)
            data["tr1"] = abs(high - close.shift())
            data["tr2"] = abs(low - close.shift())
            tr = data[["tr0", "tr1", "tr2"]].max(axis=1)
            atr = wwma(tr, n)
            return atr

        def bollinger_zscore(series: pd.Series, length: int = 20) -> pd.Series:
            # Ref: https://stackoverflow.com/a/77499303/
            rolling = series.rolling(length)
            mean = rolling.mean()
            std = rolling.std(ddof=0)
            return (series - mean) / std

        def bollinger_bands(
            series: pd.Series,
            length: int = 20,
            *,
            num_stds: tuple[float, ...] = (2, 0, -2),
            prefix: str = "",
        ) -> pd.DataFrame:
            # Ref: https://stackoverflow.com/a/74283044/
            rolling = series.rolling(length)
            bband0 = rolling.mean()
            bband_std = rolling.std(ddof=0)
            return pd.DataFrame(
                {
                    f"{prefix}{num_std}": (bband0 + (bband_std * num_std))
                    for num_std in num_stds
                }
            )

        # save data
        digits: Final = self.state.decimals(originalContract)
        table = pd.DataFrame(found).convert_dtypes()

        if table.empty:
            logger.error("No result?")
            return

        table["diff"] = table.close.diff(periods=1)

        for n in [5, 10, 20, 30, 60, 220, 325]:
            field = f"sma_{n}"
            table[field] = round(table.close.rolling(n).mean(), digits)
            table[f"{n}±"] = table.apply(
                lambda row: "+"
                if row.close > row[field]
                else "-"
                if row.close < row[field]
                else "=",
                axis=1,
            )

        table = pd.concat([table, bollinger_bands(table.close, prefix="bb_")], axis=1)
        table["bb_zscore"] = bollinger_zscore(table.close)
        table["range"] = table.high - table.low
        table["atr"] = atr(table, 6)

        table = round(table, digits)

        # saving requires removing the timezone, so keep time and just un-localize it.
        try:
            # Note: tz_convert(None) moves it to UTC. tz_localize(None) leaves the time unchanged but drops the timezone from the object.
            table.date = table.date.dt.tz_localize(None)
        except:
            # if we have a full datetime, the above works, else it fails, but failure is okay here.
            pass

        logger.info("{}", pp.pformat(contract))
        printFrame(table, f"{self.symbol} History")

        duration_map: Final = {
            "1 secs": 1,
            "5 secs": 5,
            "10 secs": 10,
            "15 secs": 15,
            "30 secs": 30,
            "1 min": 60,
            "2 mins": 120,
            "3 mins": 180,
            "5 mins": 300,
            "10 mins": 600,
            "15 mins": 900,
            "20 mins": 1200,
            "30 mins": 1800,
            "1 hour": 3600,
            "2 hours": 7200,
            "3 hours": 10800,
            "4 hours": 14400,
            "8 hours": 28800,
            "1 day": 86400,
            "1 week": 604800,
            "1 month": 2592000,
        }

        barDiffSec = duration_map[self.interval]
        logger.info("Bar length: {} ({} seconds)", self.interval, barDiffSec)

        # name output as {localSymbol}.{barDuration}.table.json
        # TODO: make the storage directory a parameter/envvar somewhere
        where = pathlib.Path("bardb")
        where.mkdir(parents=True, exist_ok=True)

        printName = self.state.nameForContract(contract).replace(" ", "_")

        # we also have to strip slashes in futures names so they don't turn into directory paths...
        filename = where / f"{printName.replace('/', ',')}.{barDiffSec}.table.json"
        table.to_json(filename, orient="table")

        # done
        logger.info("Saved to: {}", filename)

        lb = LevelBreacher(barDiffSec)

        # we store open/high/low/close values _directly_ but we use 'sma' as a prefix matcher with sub-lookback-duration fields.
        populate: Final = "open high low close average sma bb_2 bb_0 bb_-2".split()

        # use last row of dataframe (with most recently updated SMA values)
        lastlast = table.iloc[-2]
        last = table.iloc[-1]

        # create level matchers for each data field we're interested in tracking
        # (by iterating the name/field pairs of the last row created)
        levels = []

        def buildLevelsFromRow(row, desc, populateSMA=True, isToday=False):
            for name, level in row.items():
                assert isinstance(name, str)

                # if value is NaN, don't generate a level checker for this component
                if level != level:
                    continue

                for p in populate:
                    if name.startswith(p):
                        s = name.split("_")

                        # the lookback duration is EITHER a key split like 'sma_220' — OR — the direct bar interval if no sma is present.
                        # (e.g. 1 day bars with "open" tracking is "/ES DOWN open 1 day")
                        if len(s) > 1:
                            # SMA split durations are just their own size with no more details
                            lookback = int(s[1])
                            lookbackName = s[1]
                        else:
                            # else, the duration is the BAR duration, which we calculate back to whole values of minute/day/week/month/year
                            lookback = barDiffSec
                            lookbackName = convert_time(barDiffSec)

                        # SMA levels are optional because we don't want to, for example, populate *yesterday* SMA values
                        if p == "sma" and populateSMA:
                            l = LevelLevels(p, lookback, lookbackName, level)
                            levels.append(l)
                        elif p != "sma":
                            # only populate open/high/low/close/average values if we are on 30 minute bars or larger
                            # (because it's too nosiy bouncing around "last seen 2 minute close breach" etc)
                            if barDiffSec >= 1800:
                                # if bar is TODAY (meaning bar is STILL OPEN and changing close/high/low/average), then don't
                                # populate the changing fields.
                                # (Note: for TODAY we _do_ accept the current bb std ranges from the active 'close' math reported... because why not)
                                if isToday and (p != "open" and not p.startswith("bb")):
                                    continue

                                l = LevelLevels(
                                    f"{p} {desc}", lookback, lookbackName, level
                                )
                                levels.append(l)

        # date math is hacky here (isn't it always?) because last.date could be a Date _or_ a Timestamp, so we always re-wrap it in a Timestamp so the compare can work
        today = pd.Timestamp.now().floor("D")
        lastIsToday: Final = pd.Timestamp(last.date) >= today
        lastLastIsToday: Final = pd.Timestamp(lastlast.date) >= today

        # for TODAY (if row[-1] date is >= CALENDAR TODAY) we want to ignore close (random) and high/low (caught by agent-sever)
        # Note: meaning of 'today' is the last row, so 'today' is only 'today' if the stats are generated on the current trading day.
        # also, only populate the "yesterday" value if the iloc[-2] bar is NOT today.
        if barDiffSec >= 300:
            buildLevelsFromRow(
                lastlast, "previous", populateSMA=False, isToday=lastLastIsToday
            )

        buildLevelsFromRow(last, "today", isToday=lastIsToday)

        lb = LevelBreacher(barDiffSec, levels)

        symkey = lookupKey(contract)
        logger.info(
            "[{}] Created Level Breacher alerting hierarchy:\n{}",
            printName,
            pp.pformat(lb),
        )

        # a little abstraction breakage here. Is fine.
        self.state.quoteState[symkey].levels[barDiffSec] = lb


@dataclass
class IOpSimulate(IOp):
    """Simluate events (for testing event handlers mostly)"""

    category: str = field(init=False)
    params: list[str] = field(init=False)

    def argmap(self):
        return [DArg("category", convert=str.lower), DArg("*params")]

    async def run(self):
        args = dict(zip(self.params, self.params[1:]))

        match self.category:
            case "commission":
                orderId = random.randint(0, 200_000)
                qty = float(args.get("qty", 2))
                price = float(args.get("price", 6.47))
                side = args.get("side", "SLD")
                trade = Trade(
                    contract=Option(
                        conId=650581442,
                        symbol="NVDA",
                        lastTradeDateOrContractMonth="20240920",
                        strike=100.0,
                        right="P",
                        multiplier="100",
                        exchange="SMART",
                        currency="USD",
                        localSymbol="NVDA  240920P00100000",
                        tradingClass="NVDA",
                    ),
                    order=Order(
                        orderId=orderId,
                        clientId=4,
                        permId=1139889072,
                        action="BUY",
                        totalQuantity=5.0,
                        orderType="LMT",
                        lmtPrice=5.77,
                        auxPrice=0.0,
                        tif="GTC",
                        ocaGroup="1139889071",
                        ocaType=3,
                        parentId=30741,
                        displaySize=2147483647,
                        rule80A="0",
                        openClose="",
                        volatilityType=0,
                        deltaNeutralOrderType="None",
                        referencePriceType=0,
                        account="U",
                        clearingIntent="IB",
                        adjustedOrderType="None",
                        cashQty=0.0,
                        dontUseAutoPriceForHedge=True,
                    ),
                    orderStatus=OrderStatus(
                        orderId=orderId,
                        status="SIMULATED",
                        filled=0.0,
                        remaining=5.0,
                        avgFillPrice=0.0,
                        permId=1139889072,
                        parentId=0,
                        lastFillPrice=0.0,
                        clientId=4,
                        whyHeld="",
                        mktCapPrice=0.0,
                    ),
                    fills=[],
                    log=[
                        TradeLogEntry(
                            time=datetime.datetime(
                                2024,
                                8,
                                7,
                                1,
                                47,
                                27,
                                517468,
                                tzinfo=datetime.timezone.utc,
                            ),
                            status="SIMULATED",
                            message="",
                            errorCode=0,
                        ),
                        TradeLogEntry(
                            time=datetime.datetime(
                                2024,
                                8,
                                7,
                                4,
                                16,
                                13,
                                396221,
                                tzinfo=datetime.timezone.utc,
                            ),
                            status="SIMULATED",
                            message="",
                            errorCode=0,
                        ),
                    ],
                    advancedError="",
                )
                fill = Fill(
                    contract=Option(
                        conId=691069999,
                        symbol="SPX",
                        lastTradeDateOrContractMonth="20240816",
                        strike=5520.0,
                        right="P",
                        multiplier="100",
                        exchange="CBOE",
                        currency="USD",
                        localSymbol="SPXW  240816P05520000",
                        tradingClass="SPXW",
                    ),
                    execution=Execution(
                        execId="0000f711.670b89f9.02.01.01",
                        time=datetime.datetime(
                            2024, 8, 16, 10, 8, 56, tzinfo=datetime.timezone.utc
                        ),
                        acctNumber="U",
                        exchange="CBOE",
                        side=side,
                        shares=qty,
                        price=price,
                        permId=1143027176,
                        clientId=77,
                        orderId=orderId,
                        liquidation=0,
                        cumQty=2.0,
                        avgPrice=6.47,
                        orderRef="",
                        evRule="",
                        evMultiplier=0.0,
                        modelCode="",
                        lastLiquidity=1,
                        pendingPriceRevision=False,
                    ),
                    commissionReport=CommissionReport(
                        execId="",
                        commission=random.gauss(1.24, 1.24),
                        currency="",
                        realizedPNL=0.0,
                        yield_=0.0,
                        yieldRedemptionDate=0,
                    ),
                    time=datetime.datetime(
                        2024, 8, 16, 10, 8, 56, 756793, tzinfo=datetime.timezone.utc
                    ),
                )

                # test it through the live handler callback...
                self.state.commissionHandler(trade, fill, None)


@dataclass
class IOpExpand(IOp):
    """Schedule multiple commands to run based on the expansion of all inputs.

    Example:

    > expand buy {AAPL,MSFT} $10_000 AF

    Would run 2 async commands:
    > buy AAPL $10_000 AF
    > buy MSFT $10_000 AF

    Or you could even do weird things like:

    > expand buy {NVDA,AMD} {$5_000,$9_000} {AF, LIM}

    Would run all of these:
    > buy NVDA $5_000 AF
    > buy NVDA $5_000 LIM
    > buy NVDA $9_000 AF
    > buy NVDA $9_000 LIM
    > buy AMD $5_000 AF
    > buy AMD $5_000 LIM
    > buy AMD $9_000 AF
    > buy AMD $9_000 LIM

    Note: Using 'expand' for menu based commands like "limit" or "spread" will probably do weird/bad things to your interface.
    """

    parts: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*parts", desc="Command to expand then execute all combinations thereof"
            )
        ]

    async def run(self):
        # build each part needing expansion
        logger.info("Expanding request into commands: {}", " ".join(self.parts))

        assemble = [
            mutil.expand.expand_string_curly_braces(part) for part in self.parts
        ]

        # now generate all combinations of all expanded parts
        # (we break out the solution as list of [command, args] pairs)
        cmds = [(x[0], " ".join(x[1:])) for x in itertools.product(*assemble)]
        logger.info(
            "Running commands ({}): {}", len(cmds), [c[0] + " " + c[1] for c in cmds]
        )

        # now run all commands concurrently(ish) by using our standard [command, args] format to the op runner
        try:
            return await asyncio.gather(*[self.runoplive(c[0], c[1]) for c in cmds])
        except:
            logger.exception("Exception in multi-command execution?")
            return None


@dataclass
class IOpScheduleEvent(IOp):
    """Schedule a command to execute at a specific date+time in the future."""

    name: str = field(init=False)
    datetime: whenever.ZonedDateTime = field(init=False)
    cmd: list[str] = field(init=False)

    # asub /NQ COMBONQ yes 0.66 cash 15 TemaTHMAFasterSlower direct
    def argmap(self):
        return [
            DArg(
                "name",
                desc="Name of event (for listing and canceling in the future if needed)",
            ),
            DArg(
                "datetime",
                convert=lambda dt: whenever.LocalDateTime.from_py_datetime(
                    dateutil.parser.parse(dt)
                ).assume_tz("US/Eastern", disambiguate="compatible"),
                desc="Date and Time of event (timezone will be Eastern Time)",
            ),
            DArg("cmd", desc="icli command to run at the given time"),
        ]

    async def run(self):
        if self.name in self.state.scheduler:
            logger.error(
                "[{} :: {}] Can't schedule because name already scheduled!",
                self.name,
                self.cmd,
            )
            return False

        now = whenever.ZonedDateTime.now("US/Eastern")

        # "- 1 second" allows us to schedule for "now" without time slipping into the past and
        # complaining we scheduled into the past. sometimes we just want it now.
        if (now - whenever.seconds(1)) > self.datetime:
            logger.error(
                "You requested to schedule something in the past? Not scheduling."
            )
            return False

        logger.info(
            "[{} :: {} :: {}] Scheduling: {}",
            self.name,
            self.datetime,
            dict(
                zip(
                    "hours minutes seconds".split(),
                    (self.datetime - now).in_hrs_mins_secs_nanos(),
                )
            ),
            self.cmd,
        )

        async def doit() -> None:
            try:
                # "self.cmd" is the text format of a command prompt to run.
                # You can run multiple commands with standard "ls; orders; exec" syntax
                logger.info("[{} :: {}] RUNNING UR CMD!", self.name, self.cmd)
                await self.state.buildAndRun(self.cmd)
                logger.info("[{} :: {}] Completed UR CMD!", self.name, self.cmd)
            except asyncio.CancelledError:
                logger.warning(
                    "[{} :: {}] Future Scheduled Task Canceled!", self.name, self.cmd
                )
            except:
                logger.exception(
                    "[{} :: {}] Scheduled event failed?", self.name, self.cmd
                )
            finally:
                self.state.scheduler.cancel(self.name)
                logger.info("[{}] Removed scheduled event!", self.name)

        howlong = (self.datetime - now).in_seconds()
        logger.info(
            "[{} :: {}] command is scheduled to run in {:,.2f} seconds ({:,.2f} minutes)!",
            self.name,
            self.cmd,
            howlong,
            howlong / 60,
        )

        sched = self.state.scheduler.create(
            self.name,
            doit(),
            schedule=BGSchedule(
                start=self.datetime.py_datetime(), tz=pytz.timezone("US/Eastern")
            ),
            meta=self.cmd,
        )
        logger.info("[{} :: {}] Scheduled via: {}", self.name, self.cmd, sched)


@dataclass
class IOpScheduleEventList(IOp):
    """List scheduled events by name and command and target date."""

    async def run(self):
        logger.info("Listing {} scheduled events by name...", len(self.state.scheduler))
        self.state.scheduler.report()


@dataclass
class IOpScheduleEventCancel(IOp):
    """Cancel event by name."""

    name: str = field(init=False)

    def argmap(self):
        return [DArg("name", desc="Name of event to cancel")]

    async def run(self):
        got = self.state.scheduler.cancel(self.name)
        if not got:
            logger.error("[{}] Scheduled event not found?", self.name)
            return False

        logger.info("[{} :: {}] Command(s) deleted!", self.name, [g.meta for g in got])
        logger.info("[{} :: {}] Task(s) deleted!", self.name, got)


@dataclass
class IOpTaskList(IOp):
    def argmap(self):
        return []

    async def run(self):
        self.state.task_report()


@dataclass
class IOpTaskCancel(IOp):
    ids: list[int] = field(init=False)

    def argmap(self):
        return [DArg("*ids", convert=lambda xs: list(map(int, xs)))]

    async def run(self):
        for taskid in self.ids:
            logger.info("[{}] Stopping task...", taskid)
            self.state.task_stop_id(taskid)


@dataclass
class IOpAlias(IOp):
    cmd: str = field(init=False)
    args: list[str] = field(init=False)

    def argmap(self):
        return [DArg("cmd"), DArg("*args")]

    async def run(self):
        # TODO: allow aliases to read arguments and do calculations internally
        # TODO: should this just be an external parser language too?
        aliases = {
            "buy-spx": {"async": ["fast spx c :1 0 :2*"]},
            "sell-spx": {"async": ["evict SPXW* -1 0"]},
            "evict": {"async": ["evict * -1 0"]},
            "clear-quotes": {"async": ["qremove blahblah SPXW*"]},
        }

        if self.cmd not in aliases:
            logger.error("[alias {}] Not found?", self.cmd)
            logger.error("Available aliases: {}", sorted(aliases.keys()))
            return None

        cmd = aliases[self.cmd]
        logger.info("[alias {}] Running: {}", self.cmd, cmd)

        # TODO: we could make a simpler run wrapper for "run command string" instead of
        #       always breaking out the command-vs-argument strings manually.
        return await asyncio.gather(
            *[
                self.runoplive(
                    cmd.split()[0],
                    " ".join(cmd.split()[1:]),
                )
                for cmd in cmd["async"]
            ]
        )


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


@dataclass
class IOpRID(IOp):
    """Retrieve ib_insync request ID and server Next Request ID"""

    def argmap(self):
        # rid has no args!
        return []

    async def run(self):
        logger.info("CLI Request ID: {}", self.ib.client._reqIdSeq)
        logger.info(
            "Server Next Request ID: {} (see server log)", self.ib.client.reqIds(0)
        )


@dataclass
class IOpOrderModify(IOp):
    """Modify an existing order using interactive prompts."""

    def argmap(self):
        # No args, we just use interactive prompts for now
        return []

    async def run(self):
        # "openTrades" include the contract, order, and order status.
        # "openOrders" only includes the order objects with no contract or status.
        ords = self.ib.openTrades()
        # logger.info("current orders: {}", pp.pformat(ords))

        promptOrder = [
            Q(
                "Current Order",
                choices=[
                    Choice(
                        f"{o.order.action:<4} {o.order.totalQuantity:<6} -> {self.state.nameForContract(o.contract)} {o.order.orderType} {o.order.tif} lmt:${fmtPrice(o.order.lmtPrice):<7} aux:${fmtPrice(o.order.auxPrice):<7} status:{o.log[-1].status} events:{len(o.log)}",
                        o,
                    )
                    for o in sorted(
                        filter(
                            lambda x: x.orderStatus.clientId == self.state.clientId
                            and x.log[-1] != "Inactive",
                            ords,
                        ),
                        key=tradeOrderCmp,
                    )
                ],
            ),
            Q("New Limit Price"),
            Q("New Stop Price"),
            Q("New Quantity"),
        ]

        trade = None
        try:
            pord = await self.state.qask(promptOrder)

            if not pord:
                logger.info("User canceled modification request")
                return

            trade = pord["Current Order"]
            lmt = pord["New Limit Price"]
            stop = pord["New Stop Price"]
            qty = pord["New Quantity"]

            contract = trade.contract

            # start with the current live order, then COPY IT into a NEW ORDER each time
            # with per-field modifications so we aren't changing the live state (live state
            # should ONLY be updated by IBKR callbacks, not by us mutating values directly!)
            ordr = trade.order
            assert isinstance(ordr, Order)

            if not (lmt or stop or qty):
                # User didn't provide new data, so stop processing
                return None

            # TODO: if new limit price is 0, the user is trying to just do a market order, but we can't change
            #       order types, so technically we should cancel this order the create a new AMF order.

            # Collect all update parameters into a dict so we can apply them to the replaced
            update = {}
            if lmt:
                update["lmtPrice"] = Decimal(lmt)

            if stop:
                update["stop"] = Decimal(stop)

            if qty:
                # TODO: allow quantity of -1 or "ALL" to use total position?
                #       We would need to look up the current portfolio holdings to match sizes against here.
                update["totalQuantity"] = Decimal(qty)

            ordr = await self.state.safeModify(trade.contract, trade.order, **update)

            # we MUST have replaced the order by now or else the conditions above are broken
            assert ordr is not trade.order

            logger.info("Submitting order update: {} :: {}", contract, ordr)
            trade = self.ib.placeOrder(contract, ordr)
            logger.info("Updated: {}", pp.pformat(trade))
        except KeyboardInterrupt:
            if trade:
                logger.warning(
                    "[{}] Canceled update!",
                    trade.contract.localSymbol or trade.contract.symbol,
                )
        except:
            if trade:
                logger.exception(
                    "[{}] Failed to update?",
                    trade.contract.localSymbol or trade.contract.symbol,
                )

        return None


@dataclass
class IOpScaleOrder(IOp):
    """Scale-in order entry where you specify an instrument and a starting time, and it grabs prices in ±tick increments until quantity filled."""

    cmd: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*cmd",
                desc="Command in Order Lang format for buying or selling or previewing operations",
            )
        ]

    async def run(self):
        """Begin the scale-in order process for the given order command request."""
        cmd = " ".join([f"'{c}'" if " " in c else c for c in self.cmd])

        # parse the entire input string to this command through the requestlang/orderlang parser
        request = self.state.requestlang.parse(cmd)
        logger.info("[{}] Requesting: {}", cmd, request)

        self.symbol = request.symbol
        assert self.symbol
        assert request.qty

        contract = None
        self.symbol, contract = await self.state.positionalQuoteRepopulate(self.symbol)

        if not contract:
            logger.error("Contract not found for: {}", self.symbol)
            return None

        isPreview: Final = request.preview

        name = nameForContract(contract)
        tickLow = request.config.get("ticklow", 0.25)
        tickHigh = request.config.get("tickhigh", 0.25)

        assert isinstance(tickLow, (float, Decimal))
        assert isinstance(tickHigh, (float, Decimal))

        self.state.task_create(
            f"[{name}] Scale-In Automation",
            self.state.positionActiveLifecycleDoctrine(
                contract, request, tickLow, tickHigh
            ),
        )


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


@dataclass
class IOpMarketReporter(IOp):
    """Generate a periodic market strength report using internal data (emas, trade/volume, volatility) for positioning recommendations."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*symbols",
                desc="Enable periodic directional state reporting for specific symbols",
            )
        ]

    async def run(self):
        # lookup vix iticker for sharing volatility reporting everywhere
        vf, vc = await self.state.positionalQuoteRepopulate("I:VIX")
        assert vc
        vkey = lookupKey(vc)
        assert vkey
        vix = self.state.quoteState.get(vkey)
        assert vix

        async def report(iticker):
            """Report on symbol when called."""

            # report on sweeping EMA combinations (pair-wise directional directions, if all agree, major direction advantage, also scale to "age" of ticker so we don't report durations longer than the ticker has lived)
            # report on suggested instruments to use given strength
            # perhaps include dependent quotes for underlying symbol? If asking for ES direction, check our ES options for their directional strength too?
            # Difference between running this on underlying vs options for side strength?

            # Position Strength Recommendations:
            #  - DELTA-REDUCED FUTURES POSITION MULTIPLES
            #  - VOLATILITY ATM STRADDLES ON RISING VIX
            #  - DIRECTIONAL-WEIGHTED VOLATILITY STRADDLES ON LOWER VIX, RISING UNDERLYING

            age = iticker.age
            sym = iticker.contract.localSymbol

            def positionSuggestion(scores):
                vixReport = vix.rms()

                suggest = "NONE YET"

                # if vix is DECLINING, we can go directional long.
                vr300 = vixReport[300]
                if vr300 <= 0:
                    suggest = "VIX DOWN, SUGGEST DIRECTIONAL LONG"
                elif vr300 >= 0:
                    suggest = "VIX UP, SUGGEST VOLATILITY STRADDLES"

                pass

        for symbol in self.symbols:
            # Step 1: Run First Report
            # Step 2: Create reporter for symbol
            # Step 3: Schedule Continuous Reporter

            found, contract = await self.state.positionalQuoteRepopulate(symbol)

            if not found:
                logger.error("[{}] Not found?", symbol)
                continue

            assert contract

            symkey = lookupKey(contract)
            iticker = self.state.quoteState.get(symkey)

            created = self.task_create(
                f"market direction reporter for {symbol}",
                report(iticker),
                schedule=BGSchedule(
                    delay=0, runtimes=int(sys.float_info.max), pause=90
                ),
            )

            logger.info("[{}] Created recurring reporting task: {}", symbol, created)


@dataclass
class IOpAdviceMode(IOp):
    """Generate a market strength score based on active data metrics."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*symbols",
                desc="Run stats for specific symbol(s) plus default futures",
            )
        ]

    async def run(self):
        # Steps:
        #   - Check current ATM SPX strikes against their VWAP distance
        #   - VIX near HOD or LOD or neutral
        #   - Check EMA crossovers of RTY, ES, NQ for near term and long term directions
        #   - signal for "long the butt" and "short the crown"
        #   - above or below 5 minute volstop
        #   -

        # ATM SPX evaluation
        # 'straddle' returns the contracts for the added straddle/strangle bag and each leg.
        addedContracts = await self.runoplive("straddle", "SPX")

        # Remove the 'Bag' so we only have Options remaining
        spxContracts = filter(lambda x: isinstance(x, Option), addedContracts)

        # now look up tickers for each of the options...
        tickers = [self.state.quoteState[lookupKey(x)] for x in spxContracts]

        callVWAPDistance = 0
        putVWAPDistance = 0

        try:
            for t in tickers:
                match t.contract.right:
                    case "C":
                        callVWAPDistance = (t.bid - t.vwap) / t.vwap
                    case "P":
                        putVWAPDistance = (t.bid - t.vwap) / t.vwap

            logger.info("SPX Call VWAP: {:,.2f} %", callVWAPDistance * 100)
            logger.info("SPX Put VWAP: {:,.2f} %", putVWAPDistance * 100)
        except:
            logger.warning("SPX Strikes VWAP not populated yet... try again soon.")

        # VIX evaluation
        v = self.state.quoteState["VIX"]
        vixHOD = v.last - v.high
        vixLOD = v.last - v.low
        vixVWAPDistance = (v.last - v.vwap) / v.vwap

        logger.info("VIX vwap: {:,.2f} %", vixVWAPDistance * 100)

        # Index crossover checks (EMA and VWAP)
        # fetch contracts for each index
        rty, es, nq = await self.state.qualify(
            *[contractForName(x) for x in "/RTY /ES /NQ".split()]
        )

        # convert contracts to ticker lookup keys
        rtyk, esk, nqk = [lookupKey(x) for x in [rty, es, nq]]

        # fetch tickers from lookup keys
        rtyt, est, nqt = [self.state.quoteState[x] for x in [rtyk, esk, nqk]]

        def emaCheck(ticker, fast, slow) -> float:
            """Run fast/slow crossover distance for ticker and return the percentage difference."""
            return round(
                100 * (ticker.ema[fast] - ticker.ema[slow]) / ticker.ema[slow], 2
            )

        def tickerScores(ticker: ITicker):
            """Return our custom signals for a single ticker."""
            medium = emaCheck(ticker, 120, 300)
            long = emaCheck(ticker, 300, 1800)

            # current could potentially be None, so check if it exists
            current = ticker.current
            tvw = ticker.vwap

            vwap: float | None = None
            if current:
                vwap = (current - tvw) / tvw

            # get durations for EMA inside ticker, but remove the first and last entries for more stable update checks
            # (first position is just "last updated price" with no trend, and last position is approximate VWAP and doesn't change fast enough)
            durations = ticker.ema.durations[1:-1]

            # fetch emas for each duration
            emas = np.array([ticker.ema[x] for x in durations])

            # "long the butt" detector
            # idea: collect every EMA except the VWAP EMA, if mean(vals) and median(vals) are both less than the shortest and longest ema, we're in a buyable dip.
            emamedian = np.median(emas)
            emamean = np.mean(emas)
            base = (emamedian + emamean) / 2
            e0 = ticker.ema[durations[0]]
            e1 = ticker.ema[durations[-1]]
            longbutt = bool(base <= e0 and base <= e1)
            longscore = round(float(((e0 - base) + (e1 - base)) / 2), 2)

            # "short the crown" detector
            # idea: opposite of "long the butt" where we want the middle values to be higher than the more extreme values.
            shortcrown = bool(base >= e0 and base >= e1)
            shortscore = round(float(((base - e0) + (base - e1)) / 2), 2)

            # Note: for now the quoteflow reporter outputs its own prints
            qfresults = ticker.quoteflow.analyze()

            return dict(
                zip(
                    "vwap medium long butt crown buttscore crownscore quoteflow".split(),
                    [
                        vwap,
                        medium,
                        long,
                        longbutt,
                        shortcrown,
                        longscore,
                        shortscore,
                        qfresults,
                    ],
                )
            )

        # score RTY
        rtystats = tickerScores(rtyt)

        # score ES
        esstats = tickerScores(est)

        # score NQ
        nqstats = tickerScores(nqt)

        stats = dict(
            zip(
                [getattr(x, "localSymbol") for x in [rty, es, nq]],
                [rtystats, esstats, nqstats],
            )
        )

        # 5 minute volstop direction detector
        # bulk-import history for 2 days and run volstop algo for current side, duration, and distance.
        logger.info("Result: {}", pp.pformat(stats))


@dataclass
class IOpOrder(IOp):
    """Quick order entry with full order described on command line."""

    cmd: list[str] = field(init=False)

    # time in time.time()
    prevBidAskTime: float = 0
    prevBidAsk: tuple[float | None, float | None] = (None, None)

    async def currentBidAsk(
        self, contract, quoteKey
    ) -> tuple[float | None, float | None]:
        # if current quote is less than 500 ms old, return cached quote.
        # else, fetch new quote.
        # (sometimes we call currentBidAsk() multiple times on a new order,
        #  but we don't want to print the bid/ask output or actually fetch ig
        #  multiple times sequentially if we know it's "good enough" for now).
        if time.time() - self.prevBidAskTime < 0.50:
            return self.prevBidAsk

        try:
            for i in range(0, 35):
                bid, ask = self.state.currentQuote(quoteKey)

                # if we found something, we can continue
                if bid or ask:
                    self.prevBidAskTime = time.time()
                    self.prevBidAsk = (bid, ask)
                    return bid, ask

                logger.info(
                    "[{:>2} :: {}] Waiting for new quote to populate...", i, quoteKey
                )

                # else, wait for quote to populate because it could be new
                await asyncio.sleep(0.05)
        except:
            # quote isn't available currently, try again later
            logger.exception("No quote?")
            pass

        return None, None

    async def currentMidpointFor(
        self, contract, quoteKey, currentPrice: Decimal | None = None, segment=3
    ):
        """Calculate current midpoint but at 'segment' block.

        For 'segment' we divide the spread size into 10 equal sized blocks, and you can request which price edge of a block to return.

        This allows us to walk prices in more than just half increments (e.g. walk in 1/8th increments by 2,4,6,8 instead of only half increments).

        segment=2 is walking by 20%, 40%, 60%, 80% of the spread
        segment=3 is walking by 30%, 60%, 90% of the spread.
        segment=4 is walking by 40%, 80%
        segment=5 is the midpoint. (50% of the spread)
        segment=6 is walking by 60%
        segment=7 is walking by 70%
        segment=8 is walking by 80%

        We currently default to walking by segment 2 which is equivalent to using the lower 1/5th of the spread then moving inwards on each new re-pricing.
        """
        bid, ask = await self.currentBidAsk(contract, quoteKey)

        # if no bid, just go to the ask with no adjustments (because we have no valid range to determine)
        if bid is None and ask is not None:
            return Decimal(str(ask))

        if not (bid and ask):
            return None

        dbid = Decimal(str(bid))
        dask = Decimal(str(ask))

        # TODO: restore ability for OPENING ORDERS to reduce quantity when ordering and chasing quotes, but
        #       remember to DO NOT reduce quantity when chasing closing orders because we want to close a whole position.
        newPrice: Decimal | float | None
        if self.total.is_long:
            # if this is our FIRST midpoint request, we have no CURRENT PRICE, so we want a complete bid/ask midpoint
            # Also, if we are walking, and the price moves up faster than our walking, reset our estimate to the current market bid instead of our previous (now lower-than-bid) price.
            if not currentPrice or currentPrice < dbid:
                currentPrice = dbid

            # if this is a BUY LONG order, we want to close the gap between our initial price guess and the actual ask.
            width = dask - currentPrice
            newPrice = dbid + (width / 10 * segment)
        else:
            # set price if we weren't provided a price but ALSO re-align the price to the market if market jumped ahead of us during walking
            if not currentPrice or currentPrice > dask:
                currentPrice = dask

            # else, this is a SELL SHORT (or close long) order, so we want to start at our price and chase the bid closer.
            width = currentPrice - dbid
            newPrice = dask - (width / 10 * segment)

        # for now we just round everything up to the next increment. We could be smarter about rounding direction preference.
        assert newPrice is not None
        newPrice = await self.state.complyNear(contract, newPrice)
        assert newPrice is not None

        # automatically determine the "next price increment" for this contract using our instrument DB
        d = await self.state.tickIncrement(contract)
        # logger.info("Next price increment: {}", d)

        while newPrice == currentPrice:
            # TODO: if we reach at or below the bid or above the ask, maybe stop updating? We probably already filled and just didn't get a notice yet?
            # TODO: turn this update into its own object with: (bid, ask, midpoint, previousPriceAttempt, nextPriceAttempt)
            if self.total.is_long:
                # if LONG, chase UPSIDE
                newPrice = await self.state.complyUp(contract, newPrice + d)  # type: ignore
            else:
                # if short, chase DOWNSIDE
                # NOTE: DO NOT USE 'complyNear' on these because it DOES NOT CONVERGE on single repeated additions...
                newPrice = await self.state.complyDown(contract, newPrice - d)  # type: ignore

        return newPrice

    def argmap(self):
        return [
            DArg(
                "*cmd",
                desc="Command in Order Lang format for buying or selling or previewing operations",
            )
        ]

    async def run(self) -> FullOrderPlacementRecord | None | bool:
        """Run all the madness at once.

        Sure, this should probably be broken up into 5-7 different sub-functions, but it also
        kinda flows nicely as an entire novella structured in logical blocks one after another.
        """

        # we need one step of trickery here to add explicit in-line quotes to any
        # input params having spaces, so when we join them on spaces, we retain their original
        # "multi-element-quoted-value" instead of flattening *everything* and losing the
        # underlying positional context.
        cmd = " ".join([f"'{c}'" if " " in c else c for c in self.cmd])

        assert cmd, f"Why didn't you provide a buy specification to run?"

        # parse the entire input string to this command through the requestlang/orderlang parser
        request = self.state.requestlang.parse(cmd)
        logger.info("[{}] Requesting: {}", cmd, request)

        self.symbol = request.symbol
        assert self.symbol
        assert request.qty

        contract = None
        self.symbol, contract = await self.state.positionalQuoteRepopulate(
            self.symbol, request.exchange
        )

        if not contract:
            logger.error("Contract not found for: {}", self.symbol)
            return None

        # convert orderlang spec to our (maybe legacy now) PriceOrQuantity holder (until we can refactor it better to just use OrderIntent everywhere)...
        self.total = PriceOrQuantity(
            value=float(request.qty),
            is_quantity=request.isShares,
            is_money=request.isMoney,
            is_long=request.isLong,
            exchange=request.exchange,
        )

        isLong: Final = self.total.is_long

        digits: Final = self.state.decimals(contract)

        algos: Final = self.state.idb.orderTypes(contract)

        # logger.info("using contract: {}", contract)
        if contract is None:
            logger.error("Not submitting order because contract can't be formatted!")
            return None

        @functools.cache
        def qk():
            return self.state.addQuoteFromContract(contract)

        def livePrice(currentPrice=None):
            quoteKey = qk()
            return self.currentMidpointFor(contract, quoteKey, currentPrice)

        async def resolveCalculation(c):
            """Common helper for resolving calculator string replacements, calculating, then returning."""
            logger.info("Resolving calculation request: {}", c)

            # If we have a "fancy" data request, build out our entire resolver from the if-then namespace capability.
            # Basically, these are if-then resolvers prefixed with 'data:' and suffixed with the timeframe (if applicable)
            # so instead of just 'bid' you would have 'data:bid' or instead of just 'atr 300' you would have 'data:atr:300'.
            parts = c.split(" ")
            rebuild = []

            # to allow multiple data requests, we split the input on spaces to check if any are 'data-request compliant'
            # then we re-build the results back into a final calc string for processing at the end.

            # TODO: instead of all this loop extraction parsing magic here, we _could_ alter the 'calc' syntax directly
            #       to use a special data extraction operator for a symbol like (data /NQZ4 mid) or (data /NQZ4 atr 300).
            for p in parts:
                if "data:" in p:
                    partss = re.search(r"data:([^\d\s]+)(\d+)?", p)
                    entires = re.search(r"data:([^\s]+)", p)
                    if not partss and not entires:
                        logger.error(
                            "Your data field didn't match our extraction attempt: {}", c
                        )
                        return None

                    assert partss
                    assert entires
                    parts = partss.groups()
                    entire = entires.groups()[0]

                    field = parts[0].rstrip(":")
                    timeframe = int(parts[1] or 0)

                    quoteKey = qk()
                    iticker = self.state.quoteState.get(quoteKey)
                    assert iticker, f"Ticker not subscribed for {quoteKey=}?"

                    resolver = self.state.dataExtractorForTicker(
                        iticker, field, timeframe
                    )

                    # the 'entire' string _doesn't_ include the 'data:' prefix, so we add it back here for replacing the whole data request:
                    p = p.replace("data:" + entire, str(resolver()))

                    logger.info("Replaced data:{} with value: {}", entire, p)
                elif "live" in p:
                    # shorthand for what should be equal to 'data:mid'
                    currentPrice = await livePrice()
                    p = p.replace("live", str(currentPrice))
                    logger.info(
                        "Replaced live placeholder with current midpoint: {}", p
                    )

                rebuild.append(p)

            runme = " ".join(rebuild)
            logger.info("Calculating: {}", runme)
            return self.state.calc.calc(runme)

        async def configMapper() -> Mapping[str, Decimal]:
            """Convert different config names into order-required key names.

            Also convert values (potentially calculations) for each field and comply based on contract needs.
            """

            # Trailing Stop needs three values from potentially four sources:
            #  - Trailing Percent (trailingPercent) - OR - Trailing Points (aux)
            #  - Stop Price (trailStopPrice)
            #  - Distance from adjusted Trailing Stop Price for Limit to be created (lmtPriceOffset)
            RESOLVERS = dict(
                aux="aux",
                trail="aux",
                trailingpercent="trailingPercent",
                trailpct="trailingPercent",
                trailstop="trailStopPrice",
                offset="lmtPriceOffset",
                lmtoffset="lmtPriceOffset",
                lmtoff="lmtPriceOffset",
            )

            fixed = {}

            for k, v in request.config.items():
                # first, rename the key if we used an easier helper alias for the config key name
                k = RESOLVERS.get(k, k)

                # If value is a calcluation, do any replacements then run it.
                if isinstance(v, Calculation):
                    gv = await resolveCalculation(v)
                    assert isinstance(gv, Decimal)

                    logger.info("[{}] Live calculation result: {}", k, gv)
                elif isinstance(v, (Decimal, float)):
                    assert contract
                    gv = await self.state.complyNear(contract, v)
                else:
                    gv = v

                assert isinstance(gv, Decimal)
                fixed[k] = gv

            return fixed

        def verifyAlgo(algo):
            # TODO: this needs to check algo map FOR SYMBOL not for ALL OF THEM!
            # TODO: we need a better "our local names" to "underlying IBKR allowed names" like "AF" needs to map to just "LMT" for checking...
            algoValid = algo in ALGOMAP.keys()  # and (algo in orderTypes)
            if not algoValid:
                logger.error(
                    "[{}] Requested algo not found in current algos! Tried:\n{}\nAvailable total algos: {}\nAvailable order types for this instrument: {}",
                    self.symbol,
                    pp.pformat(request),
                    pp.pformat(ALGOMAP),
                    pp.pformat(sorted(algos)),
                )

                return False

            return True

        # to be used with Order().dictPopulate(config)
        config = await configMapper()

        # if qty is None, it means user requested "all" quantity, so we need to look up total current position size
        # (this only works for single symbols; spreads don't have single symbols to look up)
        if request.qty is None:
            # TODO: it would be nice if we could find "total size" for spreads too, where a spread contract has
            #       contract legs, then we can look up each leg contract matched to the ratio of the legs for
            #       the total quantity (portfolio quantity of each leg / ratio == spread quantity)
            # TODO: we can do spread size management after ordermgr is validated properly.
            contracts = self.state.contractsForPosition(self.symbol, None)

            if not contracts:
                logger.error("No contracts found for: {}", self.symbol)
                return False

            if len(contracts) > 1:
                logger.error(
                    "More than one position found? We can only order one position! Found: {}",
                    contracts,
                )
                return False

            contract, qty, delayedEstimatedMarketPrice = contracts[0]

            # an "all" request is, by usage of "all," meaning our CURRENT position for REMOVAL, so
            # an "all" quantity is the opposite of our current holding quantity.
            if qty > 0:
                # if LONG, we use SHORT SHARES to close
                request.qty = DecimalShortShares(str(qty))
            else:
                # if SHORT, we use LONG SHARES to close
                request.qty = DecimalLongShares(str(qty))

        if not verifyAlgo(request.algo):
            return False

        profitAlgo = request.bracketProfitAlgo
        lossAlgo = request.bracketLossAlgo
        if profitAlgo and not verifyAlgo(request.bracketProfitAlgo):
            return False

        if lossAlgo and not verifyAlgo(request.bracketLossAlgo):
            return False

        assert request.algo
        am = ALGOMAP[request.algo]

        isPreview: Final = request.preview

        def safeBoundsForCurrentPosition(
            maxPctChange: float,
        ) -> tuple[float, float, float]:
            """Take a float percentage (0.10) and generate (lower, upper) price bounds based on the current average position cost.

            We *ALSO* want to validate we are not buying or selling adversarial low-depth "shock quotes" so we also want to:
              - BUY no higher than 10% off the bid
              - SELL no lower than 10% off the ask

            The result values are expected to be used as:
                allowCurrentPrice = lower <= currentPrice <= upper
            """
            averagePriceSoFar = self.state.averagePriceForContract(contract)

            # convert input 0.xx percentage to multiply-growth/divide-contract percentage weight
            adjustBounds = 1 + maxPctChange

            # if we have NO position or if we are in a CLOSING MODE (short going long-to-close; or long going short-to-close),
            # then don't enforce any price caps.
            # If we want to open or close a position, we want any prices we can get at the moment
            # (assuming the bid/ask spread isn't broken; maybe we want to limit upper/lower by a wide EMA multiple of the spread midpoint).
            lower = float("-inf")
            upper = float("inf")

            # if currently holding LONG and this is still accumulating (long + long),
            # we want to limit higher prices but allow wider lower prices.
            # (if this were a closing request, it would be long + short to close)
            if averagePriceSoFar > 0 and isLong:
                upper = averagePriceSoFar * adjustBounds
            elif averagePriceSoFar < 0 and not isLong:
                # else, if currently holding SHORT and still accumulating,
                # we want to limit lower prices from crashing our cost basis,
                # but allow higher prices to increase our short profit cost basis.
                # (if this were a closing request, it would be short + long to close)
                lower = averagePriceSoFar / adjustBounds

            return lower, upper, averagePriceSoFar

        def isSafeBounds(pctBounds: float, price: float | Decimal):
            lower, upper, avgPrice = safeBoundsForCurrentPosition(pctBounds)

            safe = lower <= price <= upper
            if not safe:
                logger.error(
                    "Safe bounds ({}) exceeded: {} <= {} <= {} (with average cost: {})",
                    pctBounds,
                    lower,
                    price,
                    upper,
                    avgPrice,
                )

            return safe, avgPrice

        useAutomaticPriceWalking = False

        # begin process of sending the order to IBKR
        # if we don't have an initial price, START FROM MIDPOINT THEN WALK AROUND

        async def isSafeBidAsk(
            pctBounds: float, currentPrice: float | Decimal
        ) -> tuple[bool, float | None, float | None]:
            quoteKey = qk()
            bid, ask = await self.currentBidAsk(contract, quoteKey)

            if not (bid or ask):
                return False, None, None

            if bid is None:
                safeBid = 0.0
            else:
                safeBid = bid + (bid * pctBounds)

            if ask is None:
                return False, None, None

            safeAsk = ask - (ask * pctBounds)

            # if the bid/ask are less than the percent width apart, they overlap so we know we are within the bounds
            if safeAsk <= safeBid:
                return True, safeBid, safeAsk

            return safeBid <= currentPrice <= safeAsk, safeBid, safeAsk

        async def checkPriceSafety(
            pctBounds: float, currentPrice: float | Decimal
        ) -> bool:
            # TODO: clean up this safety check:
            #       - allow config override to disable (or change bounds?)
            #       - combine with "don't buy more than 10% above bid; don't sell more than 10% below ask" (with overrides/config too)
            assert isinstance(request.limit, (float, Decimal))
            safe, avgPrice = isSafeBounds(pctBounds, request.limit)

            if not safe:
                logger.error("Safety current position bounds failed?")
                return False

            safebidask, safebid, safeask = await isSafeBidAsk(pctBounds, request.limit)
            if not safebidask:
                logger.error(
                    "Safety check for bid/ask distance from limit failed? Got: {} <= {} <= {}",
                    safebid,
                    request.limit,
                    safeask,
                )
                return False

            return True

        # inbound order request had no price provided, which is our signal to use automatic price walking
        if request.limit is None:
            useAutomaticPriceWalking = True

            # check live quote price (hopefully populated already)
            request.limit = await livePrice()

            # if no quote found (after hours, bad symbols, IBKR maintenance times) then we can't do anything.
            if request.limit is None:
                logger.error(
                    "[{}] No live quote found. Can't create auto-generated price. Stopping order.",
                    qk(),
                )

                return None

            # TODO: make these pctBounds configurable via... some configure option
            assert isinstance(request.limit, (float, Decimal))
            safe = await checkPriceSafety(0.10, request.limit)

            if not safe:
                return False
        elif isinstance(request.limit, Calculation):
            # if the limit is a CALCULATOR TO EXECUTE, first parse it for any replacement (live prices, etc)
            # then run the calculation.
            request.limit = await resolveCalculation(request.limit)
            logger.info("[LMT] Live calculation result: {}", request.limit)

            assert isinstance(request.limit, (float, Decimal))
            safe = await checkPriceSafety(0.10, request.limit)
            if not safe:
                return False

        # limit price _must_ be a decimal here because the original parser provides decimals and we generate decimals
        assert isinstance(
            request.limit, Decimal
        ), f"Expected decimal, but got {request.limit=}?"

        multiplier = self.state.multiplier(contract)
        dmul = Decimal(str(multiplier))

        if scale := request.scale:
            scaleAvg = request.scaleAvgRecord
            assert isinstance(scaleAvg.limit, (float, Decimal))
            assert isinstance(scaleAvg.qty, (float, Decimal))

            scaleTotal = float(scaleAvg.limit) * float(scaleAvg.qty) * multiplier
            # logger.info("scale avg is: {}", scaleAvg)

            logger.info(
                "[{}] Scale Details: total qty {} @ avg cost ${:,.{}f} == total cost ${:,.{}f} ({} x {:,.{}f} x {}); profit @ {:,.{}f} (+{} % :: ${:,.{}f}); loss @ {:,.{}f} (-{} % :: ${:,.{}f})",
                contract.localSymbol or contract.symbol,
                scaleAvg.qty,
                scaleAvg.limit,
                digits,
                scaleTotal,
                digits,
                scaleAvg.qty,
                scaleAvg.limit,
                digits,
                multiplier,
                scaleAvg.bracketProfitReal or nan,
                digits,
                scaleAvg.bracketProfit,
                ((scaleAvg.bracketProfitReal - scaleAvg.limit) * scaleAvg.qty * dmul)
                if scaleAvg.bracketProfitReal
                else nan,
                digits,
                scaleAvg.bracketLossReal or nan,
                digits,
                scaleAvg.bracketLoss,
                ((scaleAvg.limit - scaleAvg.bracketLossReal) * scaleAvg.qty * dmul)
                if scaleAvg.bracketLossReal
                else nan,
                digits,
            )

        async def updateBracket() -> Bracket | None:
            """Update bracket using current state.

            We make this a local closure because during walking order modification, we move the bracket prices
            by overwriting 'request.limit' then re-running this to calculate the new bracket offsets.
            """

            bracket = None
            assert contract is not None

            profitAlgo = request.bracketProfitAlgo
            lossAlgo = request.bracketLossAlgo

            if request.bracketProfit:
                # Create PROFIT exit at +$A.BC PROFIT
                # (profit is ABOVE your entry for a long, but BELOW your entry for a short.
                #  You desginate your PROFIT in +points in either case and we handle the sign flip automatically)

                # If value is a calcluation, do any replacements then run it.
                if isinstance(request.bracketProfit, Calculation):
                    request.bracketProfit = DecimalPrice(
                        await resolveCalculation(request.bracketProfit)
                    )
                    logger.info(
                        "[{}] Live calculation result: {}",
                        "bracketProfit",
                        request.bracketProfit,
                    )

                assert request.bracketProfitReal is not None

                extensionPriceProfit = await self.state.complyNear(
                    contract, request.bracketProfitReal
                )

                profitAlgo = ALGOMAP[profitAlgo]  # type: ignore

                # We don't need a check for an existing bracket define here
                # because this is the first bracket check case, so always create
                # if we reach here.
                bracket = Bracket(
                    profitLimit=extensionPriceProfit, orderProfit=profitAlgo
                )

            if request.bracketLoss:
                # Create LOSS exit at -$X.YZ LOSS
                # (even if short, the loss value here will be YOUR loss so ABOVE your entry)

                # If value is a calcluation, do any replacements then run it.
                if isinstance(request.bracketLoss, Calculation):
                    request.bracketLoss = DecimalPrice(
                        await resolveCalculation(request.bracketLoss)
                    )
                    logger.info(
                        "[{}] Live calculation result: {}",
                        "bracketLoss",
                        request.bracketLoss,
                    )

                assert isinstance(request.bracketLossReal, (float, Decimal))
                extensionPriceLoss = await self.state.complyNear(
                    contract, request.bracketLossReal
                )

                lossAlgo = ALGOMAP[lossAlgo]  # type: ignore

                # TODO: we should be able to configure an explicit stop activation (aux) vs. stop limit price
                #       if we are using stop limit orders for loss triggering...
                #       (right now we just default the limit price to the stop price which may not always work)
                #       (or we could make this more adaptive with trailing stop limit orders instead)
                if not bracket:
                    bracket = Bracket(lossLimit=extensionPriceLoss, orderLoss=lossAlgo)
                else:
                    # NOTE: we aren't currently consuming the 'aux' config value INTO the request object for bracket math.
                    # So, your stop 'aux' level is not adjusted here after your entry definition.

                    # else, add to existing bracket from profit entry above
                    bracket.lossLimit = extensionPriceLoss
                    bracket.lossStop = extensionPriceLoss
                    bracket.orderLoss = lossAlgo

            return bracket

        bracket = await updateBracket()

        # BUY CMD
        assert self.symbol
        placed = await self.state.placeOrderForContract(
            self.symbol,
            isLong,
            contract,
            qty=self.total,
            # if we are buying by AMOUNT, we don't specify a limit price since it will be calculated automatically,
            # then we read the calculated amount to run the auto-price-tracking attempts.
            # if we provide a price, it gets used as the limit price directly.
            limit=request.limit,
            orderType=am,
            preview=isPreview,
            bracket=bracket,
            config=config,
            ladder=Ladder.fromOrderIntent(request),
        )

        if isPreview:
            # Don't continue trying to update the order if this was just a preview request
            return placed

        if not placed:
            logger.error("[{}] Order can't continue!", self.symbol)
            return False

        # if this is a market order, don't run the algo loop
        if {
            "MOO",
            "MOC",
            "MKT",
            "LIT",
            "MIT",
            "SLOW",
            "PEG",
            "STP",
            "STOP",
            "MTL",
        } & set(am.split()):
            logger.warning(
                "Not running price algo because this is a passive or slower resting order..."
            )
            return False

        # Also don't run the algo loop if we have a MANUAL price requested:
        if not useAutomaticPriceWalking:
            logger.warning(
                "Not running price algo because limit price provided directly..."
            )
            return False

        # If we reach here, we are letting the following code LIVE UPDATE THE PRICE UNTIL ALL QTY IS FILLED

        # Extract fields from the return value of the order placement
        profitTrade = None
        profitOrder = None
        lossTrade = None
        lossOrder = None
        match placed:
            case FullOrderPlacementRecord(
                limit=TradeOrder(trade=trade, order=order),
                profit=profitRecord,
                loss=lossRecord,
            ):
                if profitRecord:
                    profitTrade = profitRecord.trade
                    profitOrder = profitRecord.order

                if lossRecord:
                    lossTrade = lossRecord.trade
                    lossOrder = lossRecord.order

        # register a CUSTOM status event handler into the trade status system so even if we aren't listening
        # during a live event emission, we still have a record "something updated" for us to check.
        event = asyncio.Event()

        # record our previous status in a dict so the closure can modify persistent state outside of itself
        statusState = dict(
            prevStatus=trade.orderStatus.status,
            prevRemaining=trade.orderStatus.remaining,
            prevMessage=trade.log[-1].message,
        )

        logger.info("Starting with trade status: {}", statusState)

        def statusUpdate(tr):
            # we need to declare 'statusState' as nonlocal because this is being called from very far away...
            # (and right now the event system doesn't have a way to pass state objects into callbacks)
            nonlocal statusState

            # Only update our event notification if something useful changed.
            # (e.g. don't updated on status going from Submitted -> Submitted each modification price update)
            logger.info("Got status update: {}", tr.orderStatus)

            if (
                tr.orderStatus.status != "ValidationError"
                and tr.orderStatus.status != statusState["prevStatus"]
            ):
                logger.info(
                    "[{} -> {}] Status updated!",
                    statusState["prevStatus"],
                    tr.orderStatus.status,
                )
                event.set()
            else:
                logger.info(
                    "[{} -> {}] Not triggering status update event because order hasn't made market progress",
                    tr.orderStatus.status,
                    statusState["prevStatus"],
                )

            message = tr.log[-1].message
            if message and message != statusState["prevMessage"]:
                logger.info(
                    "[{} :: {}] New message: {}",
                    trade.order.orderId,
                    trade.orderStatus.status,
                    message,
                )

            # update history with current state to compare against next update
            statusState |= dict(
                prevStatus=tr.orderStatus.status,
                prevRemaining=tr.orderStatus.remaining,
                prevMessage=message,
            )

        trade.statusEvent += statusUpdate

        started: Final = time.time()

        def howlong():
            """Return number of seconds since we started running this order update process.

            This isn't the exact time since the order was _placed_ since the trade is already
            working in the IBKR system by now, but this is close enough for our update purposes.
            """
            return time.time() - started

        # TODO: make price-walking wait interval more configurable.
        # We want to give every price update a _chance_ to work before smashing with new price updates.
        # TODO: make WAIT DURATION adaptive or a config parameter (futures wait less than options, etc)
        PRICE_UPDATE_WAIT_DURATION = 1.25
        MESSAGE_UPDATE_DURATION = 0.222

        # when running price updates, run when (now - previous update >= PRICE_UPDATE_WAIT_DURATION) checked every MESSAGE_UPDATE_DURATION
        # TODO: we could also make this an adaptable timer difference too I guess
        PREVIOUS_UPDATE_AT = 0.0

        def howlongevent() -> float:
            """Time elapsed between now and the previous trigger time"""
            if PREVIOUS_UPDATE_AT:
                return time.time() - PREVIOUS_UPDATE_AT

            # if no recorded update time yet, we don't have a valid duration to measure against
            return np.nan

        def readyForNextUpdateTrigger() -> bool:
            # the trigger is only valid if the previously set time is populated
            if PREVIOUS_UPDATE_AT:
                return howlongevent() >= PRICE_UPDATE_WAIT_DURATION

            return False

        def nextTriggerDuration() -> float:
            """Next time to run a price check in seconds based on our delay/wait/retry paraemters and the previous time attempt."""

            # if we have a previous update time, then the logic holds
            # (if _more_ time has passed than our duration, then the check is negative, and we return 0 for "CHECK NOW",
            #  otherwise if _less_ time has passed than our duration, then the time is "padded up to" our duration check interval again)
            if PREVIOUS_UPDATE_AT:
                return max(
                    0, PRICE_UPDATE_WAIT_DURATION - (time.time() - PREVIOUS_UPDATE_AT)
                )

            # if we don't have a previous update time, then our time to wait us the duration itself at most
            return PRICE_UPDATE_WAIT_DURATION

        # TODO: refactor all of this into a self-contained price tracking agent system...
        while not trade.isDone():
            ## BEGIN AUTOMATIC PRICE WALKING LOOP STAGE (waiting for status changes, running price updates until filled or stopped) ##
            # wait for the order to go from PreSubmitted -> Submitted.
            # It doesn't make sense to start adjusting prices until the order is actually interacting with live spreads.
            # OUTER LOOP IS FOR THE ENTIRE PRICE MANAGEMENT SYSTEM TO RUN UNTIL ALL FILLS ARE COMPLETE
            while not trade.isDone():
                # record if order is currently active on an exchange *before* the next update triggers.
                # If order is NOT active (i.e. NEW order) and it BECOMES active, we do not run price updates on the first "Submitted/active" notification
                # because we want to give the order a couple seconds to try and hit an offer instead of updating the price 10 ms after it goes live.
                currentlyWorking = trade.isWorking()

                # INNER LOOP IS ONLY WAITING FOR NEXT EVENT TO TRIGGER BEFORE CHECKING STATUS AND FILLS AND PRICES AGAIN
                while not trade.isDone():
                    try:
                        timeout = min(MESSAGE_UPDATE_DURATION, nextTriggerDuration())
                        logger.info(
                            "[{} :: {} :: lmt ${:,.{}f} :: {:,.2f} seconds since last event; {:,.2f} seconds total] Waiting {:,.2f} s for next update...",
                            trade.order.orderId,
                            trade.orderStatus.status,
                            trade.order.lmtPrice,
                            digits,
                            howlongevent(),
                            howlong(),
                            timeout,
                        )

                        # use the API event system to wait for a live event to happen.
                        # possible events are: statusEvent modifyEvent fillEvent commissionReportEvent filledEvent cancelEvent cancelledEvent

                        # if we get a status update, _something_ happened (either a status transition or a full fill a partial fill or a cancel)
                        await asyncio.wait_for(event.wait(), timeout=timeout)
                        event.clear()

                        logger.info(
                            "[{} :: {} :: {} of {}] Received status update in wait cycle... checking for completeness...",
                            trade.order.orderId,
                            trade.orderStatus.status,
                            trade.orderStatus.filled,
                            trade.orderStatus.total,
                        )

                        break
                    except (KeyboardInterrupt, asyncio.CancelledError):
                        # catches CTRL-C during sleep
                        logger.warning(
                            "[{}] User canceled waiting before order checks completed! Order still live.",
                            trade.orderStatus.orderId,
                        )
                        return False
                    except asyncio.TimeoutError:
                        # catch wait_for() timeout
                        # If we reached the next update trigger time due to timeouts (of not receiving status updates),
                        # then we need to break this loop and start updating some prices...
                        if readyForNextUpdateTrigger():
                            logger.info("Ready for next price update check...")
                            break

                        # else, this was just a timeout for us to print the next user status log line, so continue
                        continue
                    except:
                        logger.exception("Something else broke?")

                if trade.isDone():
                    logger.info(
                        "[{} :: {} :: {} of {}] Order is marked as complete! Not running further price updates.",
                        trade.order.orderId,
                        trade.orderStatus.status,
                        trade.orderStatus.filled,
                        trade.orderStatus.total,
                    )
                    break

                if trade.isWaiting():
                    logger.info(
                        "[{} :: {} :: {} of {}] Order is marked as not live yet! Not running price updates until order goes live.",
                        trade.order.orderId,
                        trade.orderStatus.status,
                        trade.orderStatus.filled,
                        trade.orderStatus.total,
                    )
                    continue

                # At this point, all we know is: the order is CURRENTLY LIVE and CURRENTLY WORKING at an exchange

                # defensive logic check. This should _never_ trigger, but if it does, it means you were
                # about to get stuck in a zero-millisecond infinite loop, so here you have a chance to
                # manually cancel your operations.
                TIME_CHECK = time.time()
                if abs(TIME_CHECK - PREVIOUS_UPDATE_AT) <= 0.010:
                    logger.error(
                        "We don't support updates this fast! Attempted loop after only {} - {} = {} seconds. Pausing to continue...",
                        TIME_CHECK,
                        PREVIOUS_UPDATE_AT,
                        TIME_CHECK - PREVIOUS_UPDATE_AT,
                    )
                    try:
                        await asyncio.sleep(0.333)
                    except:
                        logger.error(
                            "User stooped live order update tracking. Order is still active."
                        )
                        break

                # since we fell out of the event wait loop, we MUST mark this as our previous update attempt, so the next
                # time the logger triggers an update, the wait exception catch knows to not continue price activity
                # until at least PRICE_UPDATE_WAIT_DURATION has passed.
                PREVIOUS_UPDATE_AT = time.time()

                # if this is our FIRST notification of the status moving to an exchange, we reset the timer waiting
                # for our first live fill (we don't include the "pre-submitted" phase as part of our walk-wait timer).
                if (not currentlyWorking) and trade.isWorking():
                    # TODO: *should* we be targeting a NEW midpoint here if the price moved between when we placed
                    #       the order and when the order hit the exchange (can be 1-5 seconds sometimes?)
                    if trade.orderStatus.status != "Submitted":
                        logger.warning("BUG: Why isn't status Submitted here?")

                    logger.info(
                        "[{} :: {} :: {} of {}] Received first Submitted status update... pausing {} seconds to wait for live fills now...",
                        trade.order.orderId,
                        trade.orderStatus.status,
                        trade.orderStatus.filled,
                        trade.orderStatus.total,
                        PRICE_UPDATE_WAIT_DURATION,
                    )
                    # the next price update will happen in PRICE_UPDATE_WAIT_DURATION seconds from now
                    # (if the order doesn't fill before then)
                    continue

                # if order is COMPLETE then STOP TRYING
                if trade.orderStatus.remaining == 0:
                    logger.error(
                        "How is remaining quantity zero here if the order is still live?"
                    )
                    return True

                # At this point, we know:
                #  - the order is LIVE AND WORKING
                #  - the order has been LIVE for a minimum of PRICE_UPDATE_WAIT_DURATION seconds
                #  - the order IS NOT COMPLETE YET
                #  - so we UDPATE PRICES CLOSER TO THE NBBO

                # convert trade object order price to Decimal() so our math doesn't break due to type issues
                currentPrice = Decimal(str(trade.order.lmtPrice))

                # re-read current quote for symbol before each update in case the price is moving rapidly against us
                logger.info("Adjusting price for more aggressive fills...")

                if (newPrice := await livePrice(currentPrice)) is None:
                    logger.error(
                        "Failed to find new midpoint for price updating! Order still active."
                    )
                    return False

                if trade.isDone():
                    logger.warning(
                        "Trade completed while checking live price! Not updating anymore."
                    )
                    return False

                # when using automatic price walking, don't increase our cost basis by more than 10%
                # (this prevents things like if we have a cost basis of $3 and are walking up, but something
                #  suddenly offers $5 instead of $3.10, we don't grab $5 suddenly)
                # TODO: consume price walk limit as a config parameter from the order intent too
                safe, avgPrice = isSafeBounds(0.10, newPrice)
                if not safe:
                    # the isSafeBounds() reports its own error message
                    continue

                # TODO: inject logic here to, if this is a current position, don't move our cost basis against us by more than 5%? 10%?
                #       Basically: cap the maximum we can modify the price here.
                #       ALSO, this needs more accurate "broken spread" checks where if the spread is greater than 50% we don't trust it.
                #       ALSO, track the spread width itself and if it bounces from "normal" to weird/wide, maybe bail out or just stop for a while.

                logger.info(
                    "Price changing from {:,.{}f} to {:,.{}f} (difference {:,.{}f}) for spending {:,.{}f}",
                    currentPrice,
                    digits,
                    newPrice,
                    digits,
                    newPrice - currentPrice,
                    digits,
                    (float(newPrice) * trade.orderStatus.remaining),
                    digits,
                )

                if currentPrice == newPrice:
                    logger.error(
                        "Not submitting order update because no price change? Order still active!"
                    )
                    return False

                # generate NEW ORDER OBJECTS with ACCEPTABLE API PARAMETERS or else IBKR rejects updates
                request.limit = newPrice
                if bracket:
                    bracket = await updateBracket()
                    assert bracket

                    # UPDATE BRACKET IF NECESSARY
                    if profitTrade:
                        logger.info(
                            "Updating PROFIT exit: ${:,.{}f}",
                            bracket.profitLimit,
                            digits,
                        )
                        profitTrade.order = await self.state.safeModify(
                            profitTrade.contract,
                            profitTrade.order,
                            lmtPrice=bracket.profitLimit,
                        )

                    if lossTrade:
                        logger.info(
                            "Updating LOSS exit: ${:,.{}f}", bracket.lossLimit, digits
                        )
                        lossTrade.order = await self.state.safeModify(
                            lossTrade.contract,
                            lossTrade.order,
                            lmtPrice=bracket.lossLimit,
                            aux=bracket.lossStop,
                        )

                updatedOrder = await self.state.safeModify(
                    trade.contract, trade.order, lmtPrice=newPrice
                )

                if trade.isDone():
                    logger.warning(
                        "Trade completed while generating modified order! Not updating anymore."
                    )
                    return False

                logger.info(
                    "[{} :: {} :: {} of {}] Submitting order update to: {} Using: {}",
                    trade.order.orderId,
                    trade.orderStatus.status,
                    trade.orderStatus.filled,
                    trade.orderStatus.total,
                    newPrice,
                    pp.pformat(updatedOrder),
                )

                # submit our PRICE MODIFICATION to IBKR for updating our existing order on the exchange(s)
                # NOTE: we update brackets (if any) BEFORE updating the order price so we maintain the expected ± offsets from the original order.
                if profitTrade:
                    self.ib.placeOrder(profitTrade.contract, profitTrade.order)

                if lossTrade:
                    self.ib.placeOrder(lossTrade.contract, lossTrade.order)

                self.ib.placeOrder(trade.contract, updatedOrder)

        # for now, stop trying to automatically run "positions" because it was too delayed and confusing us a bit
        if False:
            # now just print current holdings so we have a clean view of what we just transacted
            # (but schedule it for a next run so so the event loop has a chance to update holdings first)
            async def delayShowPositions():
                await self.runoplive("positions", [])

            self.task_create(
                "show ord positions",
                delayShowPositions,
                schedule=BGSchedule(delay=0.77, runtimes=3, pause=0.33),
            )

        # "return True" signals to an algo caller the order is finalized without errors.
        return True


@dataclass
class IOpStraddleQuote(IOp):
    """Generate a long straddle/strangle quote by providing symbol and width in strikes from ATM for put/call strikes (width 0 is just an ATM(ish) straddle).

    Note: ONLY adds quotes. DOES NOT place orders. You need to run your own orders from the quotes if necessary.


    Command supports multiple operating modes via custom parsing of the "widths" list:

    - straddle AAPL 0 — generate put/call straddle quote ATM
    - straddle AAPL 0 -10 — generate an iron condor with long put/call ATM then short put/call 10 points further out on each side.
    - straddle AAPL vertical call 0 10 — generate only a vertical call spread buying ATM+0 and selling +10 ATM
    - straddle AAPL vertical put -5 -10 - generate only a vertical put spread buying ATM-5 and selling ATM-10 ATM

    Shorthand can also be used:

    - straddle AAPL v c 2 4
    - straddle AAPL v p -3 -6

    You can also combine verticals because the inputs are consumed by just alternating reads:

    - strad /ES v c 10 20 v p -10 -20 — generates a combined call spread ATM+10 10 wide and a put spread ATM-10 10 wide.

    Note:
      - all offset math is calculated FROM THE CURRENT ATM STRIKE. So a 10 point wide spread is "0 10" or "10 20"
      - The "regular" straddle command takes sign into account, so spreads require alternating positve (long) and negative (short)
      - The "vertical" mode doesn't take sign into account for long/short since it knows the closer strike is long and the further strike is short.
    """

    symbol: str = ""
    widths: list[str] = field(default_factory=list)
    tradingClass: str = field(init=False)

    # Optionally allow using this command externally with a non-ATM automatic price starting position.
    overrideATM: float = 0.0

    def argmap(self):
        return [
            DArg(
                "symbol",
                convert=lambda x: x.upper(),
                verify=lambda x: (" " not in x) and not x.isnumeric(),
                desc="Underlying for contracts to use for ATM quote and leg choices (we auto-convert SPX->SPXW and NDX->NDXP)",
            ),
            # TODO: convert this to another sub-parser. for now we're doing hacks again.
            DArg(
                "*widths",
                default=["0"],
                desc="How many points apart to place your legs (can be 0 to just to strangle P/C ATM). Multiple widths can be defined. Positive widths are long and negative widths are short (all widths are priced in points away from ATM).",
            ),
        ]

    async def run(self):
        # Step 1: resolve input symbol to a contract
        # Step 2: fetch chains for underlying
        # Step 3: fetch current price for underlying
        # Step 4: Determine strikes to use for straddle
        # Step 5: build spread
        # Step 6: add quote for open spread
        # Step 6: add quote for close spread
        # Step 7: place spread or preview calculations

        # note: we 'convert' here because we call this class externally without the DArg automation
        if not self.widths:
            self.widths = ["0"]

        self.widths = list(map(str.upper, self.widths))

        # TODO: also replace this with the command handler, and update common handler to handle I: additions?
        if self.symbol[0] == ":":
            found, contract = self.state.quoteResolve(self.symbol)
            assert found

            self.symbol = found
            tradingClass = ""
        else:
            # Hacky way to provide trading class as symbol input...
            if "-" in self.symbol:
                self.symbol, tradingClass = self.symbol.split("-")
            else:
                tradingClass = ""

            # logger.info("[{}] Looking up underlying contract...", self.symbol)

            # we need to hack the symbol lookup if these are index options...
            prefix = "I:" if self.symbol in {"SPX", "NQ", "RTY", "VX", "VIX"} else ""
            contract = contractForName(f"{prefix}{self.symbol}")

            logger.info("[{}{}] Looking up: {}", prefix, self.symbol, contract)
            (contract,) = await self.state.qualify(contract)

            # Remove localSymbol because it conflicts with date placement on futures and the discovery
            # system prefers to use symbol and expiration date anyway.
            contract.localSymbol = ""

            logger.info("[{}] Found contract: {}", self.symbol, contract)
            assert contract.conId, f"Why isn't contract qualified? {contract=}"

        if not isinstance(contract, (Stock, Future, Index)):
            logger.error(
                "Spreads are only supported on stocks and futures and index options, but you tried to run a spread on: {}",
                contract,
            )
            return

        # TODO: cleanup this confusion between tradeSymbol and quotesym. Are they practically the same thing?
        tradeSymbol = contract.symbol

        # also hack here for index contracts having 'I:' prefix but we don't want it for the quote lookup.
        try:
            quotesym = lookupKey(contract)
        except:
            logger.exception("failed here?")
            # if lookup fails, we need to add the quote then try again
            await self.runoplive("add", tradeSymbol)

        # Convert different underlying symbols than trade symbols.
        if tradeSymbol == "SPX":
            tradeSymbol = "SPXW"
        elif tradeSymbol == "NDX":
            tradeSymbol = "NDXP"
        elif tradeSymbol == "VIX":
            tradeSymbol = "VIXW"
        elif isinstance(contract, Future):
            tradeSymbol = "/" + tradeSymbol

        try:
            for i in range(0, 100):
                if quote := self.state.quoteState.get(quotesym):
                    break

                logger.info(
                    "[{} :: {}] Quote doesn't exist yet, adding quote...",
                    tradeSymbol,
                    quotesym,
                )
                await self.runoplive("add", tradeSymbol)
                await asyncio.sleep(0)
        except:
            logger.exception("Failed to find symbol for adding quotes!")
            return False

        if not quote:
            logger.error("Quote didn't work? Can't do this without a quote.")
            return False

        # don't buy Stock(ES) instead of Future(ES)
        if isinstance(contract, Future):
            quotesym = "/" + contract.symbol

        strikes = await self.runoplive("chains", quotesym)
        strikes = strikes[quotesym]
        # logger.info("[{}] Got strikes: {}", quotesym, strikes)

        if not strikes:
            logger.error("[{} :: {}] No strikes found?", self.symbol, quotesym)
            return None

        # if we have a magic value requesting a "fake" atm for spread calculation, use it directly...
        if self.overrideATM:
            currentPrice = self.overrideATM
        else:
            # TODO: if after hours and quoting SPX, use SPX, use more dynamic ES offset.
            if quote.bid is None or quote.ask is None:
                if quote.last is None:
                    logger.error(
                        "No prices are populated, so we can't find a market price!"
                    )
                    return None

                # don't complain about an index not having bid/ask quotes
                if not isinstance(contract, Index):
                    logger.warning(
                        "[{}] Quotes aren't populated, so using LAST price",
                        contract.symbol,
                    )
                currentPrice = quote.last
            else:
                currentPrice = (quote.bid + quote.ask) / 2

        # for futures, round to nearest 10 mark because the liquidity for futures options
        # on the 5 increments is worse than the round 10 increments.
        # TODO: when we replace this with a full grammar, make round10 vs nearest-exact
        #       an in-line config option.
        if (
            (roundto := self.state.localvars.get("roundto"))
            and isinstance(contract, Future)
            and currentPrice > 1_000
        ):
            currentPrice = float(roundnear(int(roundto), currentPrice))

        logger.info("[{}] Using ATM price: {:,.2f}", self.symbol, currentPrice)

        logger.info(
            "Expiration dates ({}): {}",
            len(strikes),
            ", ".join(sorted(strikes.keys())),
        )
        sstrikes = sorted(strikes.keys())

        now = whenever.ZonedDateTime.now("US/Eastern")

        # if after market close, use next day
        if (now.hour, now.minute) >= (16, 0):
            now = now.add(days=1, disambiguate="compatible")

        # Note: DTE requests are in CALENDAR DAYS and not MARKET DAYS.
        # e.g. setting 5 DTE on a Friday for daily expirations gives you a
        #      Wednesday expiration (1: Sat, 2: Sun, 3: Mon, 4: Tues, 5: Wed)
        #      instead of 5 market days (which would be the next Friday).
        dte = int(self.state.localvars.get("dte", 0))
        now = now.add(whenever.days(dte), disambiguate="compatible")

        # Note: IBKR Expiration formats in the dict are *full* YYYYMMDD, not OCC YYMMDD.
        expTry = now.date()

        expirationFmt = f"{expTry.year}{expTry.month:02}{expTry.day:02}"
        useExpIdx = bisect.bisect_left(sstrikes, expirationFmt)

        # TODO: add param for expiration days away or keep fixed?
        self.expirationAway = 0
        useExp = sstrikes[useExpIdx : useExpIdx + 1 + self.expirationAway][-1]
        logger.info(
            "Using start date {} to discover nearest expiration ({} DTE): {}",
            expTry,
            dte,
            useExp,
        )

        assert useExp in strikes
        useChain = strikes[useExp]

        legs = []

        currentStrikeIdx = find_nearest(useChain, currentPrice)
        atm = useChain[currentStrikeIdx]

        type Strategy = Literal["STRADDLE", "STRANGLE", "VERTICAL", "VERT", "S", "V"]
        type PutCall = Literal["PUT", "CALL", "P", "C", "PS", "CS"]

        currentStrategy: Strategy = "STRADDLE"

        currentSide: PutCall | None = None
        currentlyLong: bool = False

        prewidth: Strategy | PutCall | str
        for prewidth in self.widths:
            # if the width designator is a mode switch, switch modes then continue
            if prewidth in {"STRADDLE", "STRANGLE", "VERTICAL", "VERT", "S", "V"}:
                currentStrategy = prewidth  # type: ignore
                currentSide = None
                continue

            match currentStrategy:
                case "STRADDLE" | "STRANGLE" | "S":
                    assert not prewidth.isalpha()
                    width = float(prewidth)

                    # reset the vert strategy state each time here in case we switch back over
                    currentSide = None
                    currentlyLong = False

                    # Get the SIGN of the input width.
                    # This is a bit weird because we _want_ to allow '-0' for short straddle/strangles, but
                    # there's no other way to detect -0 without stealing the sign of the input and placing it
                    # on another number to compare against.
                    # Copy sign of 'self.width' onto 1 then see if 1 is positive.
                    widthIsPositive = math.copysign(1, width) > 0

                    # allow us to build short spreads too to potentially combine with long spreads for iron condors
                    # if 'buildOpposite' is true (negative width) then we generate sells instead of buys
                    buildLong = widthIsPositive
                    width = abs(width)

                    # nearest ATM strike to the current quote
                    currentStrikeIdxLower = find_nearest(useChain, currentPrice - width)
                    currentStrikeIdxHigher = find_nearest(
                        useChain, currentPrice + width
                    )

                    put = useChain[currentStrikeIdxLower]
                    call = useChain[currentStrikeIdxHigher]
                    assert put <= atm <= call

                    side = "BUY" if buildLong else "SELL"
                    legs.append((side, "P", put))
                    legs.append((side, "C", call))
                case "VERTICAL" | "VERT" | "V":
                    # alternate adding long and short sides vertically. We also need to consume the put/call indicator.
                    if not currentSide:
                        assert prewidth in {"PUT", "CALL", "P", "C", "PS", "CS"}
                        currentSide = prewidth  # type: ignore
                        continue

                    assert not prewidth.isalpha()
                    width = float(prewidth)

                    currentStrikeIdx = find_nearest(useChain, currentPrice + width)
                    strike = useChain[currentStrikeIdx]
                    legs.append(
                        ("BUY" if not currentlyLong else "SELL", currentSide[0], strike)
                    )

                    # if this is a "safe put/call spread" add the extra protection leg on the SELL entry...
                    # (adding on SELL entry so this flips to a BUY when the spread is shorted)
                    if currentlyLong and currentSide in {"PS", "CS"}:
                        # wider protection for puts than calls because of skew
                        currentStrikeIdx = find_nearest(
                            useChain,
                            currentPrice
                            + (width * (2 if currentSide[0] == "P" else 1.5)),
                        )
                        strike = useChain[currentStrikeIdx]
                        legs.append(("SELL", currentSide[0], strike))

                    # flip long/short/long/short for the vertical spread buy/sell alternating
                    currentlyLong = not currentlyLong

        # only print this summary if we have a simple spread
        # TODO: abstract this out to show full metrics for larger inputs
        if currentStrategy == "STRADDLE" and len(legs) == 2:
            logger.info(
                "[{} :: {}] USING SPREAD: [put {:.2f}] <-(${:,.2f} wide)-> [atm {:.2f}] <-(${:,.2f} wide)-> [call {:.2f}]",
                self.symbol,
                tradeSymbol,
                put,
                atm - put,
                atm,
                call - atm,
                call,
            )

        # always create them in a consistent order
        legs = sorted(legs)

        logger.info("Legs Report: {}", pp.pformat(legs))
        if not legs:
            logger.error("No legs generated? What went wrong?")
            return None

        tradingClassExtension = f"-{tradingClass}" if tradingClass else ""
        spread = " ".join(
            [
                f"{buyorsell} 1 {tradeSymbol}{useExp[2:]}{direction}{int(strike * 1000):08}{tradingClassExtension}"
                for buyorsell, direction, strike in legs
            ]
        )

        # We don't need to manually parse the order request here since we're using `add` directly again.
        # orderReq = self.state.ol.parse(spread)

        # subscribe to quote for spread so we can figure out the current price for ordering...
        # Note: when adding spreads, 'add' also adds quote for each leg individually too.
        addedContracts = await self.runoplive("add", f'"{spread}"')

        return addedContracts


@dataclass
class IOpOrderFast(IOp):
    """Place a momentum order for scalping using multiple strikes and active price tracking.

    For a 'symbol' at total dollar spend of 'amount' and 'direction'.

    For ATR calcuation, requires a custom endpoint running with a REST API capable of
    returning multiple ATR periods for any symbol going back multiple timeframes.

    Maximum strikes attempted are based on the recent ATR for the underlying,
    so we don't try to grab a +$50 OTM strike with recent historical movement
    was only $2.50. (TODO: if running against, 2-4 week out chains allow higher
    than ATR maximum because the vol will smile).

    The buying algo is:
        - use current underlying bid/ask midpoint (requires live quote)
        - set price cap to maximum of 3 day or 20 day ATR (requires external API)
        - buy 1 ATM strike (requires fetching near-term chain for underlying)
        - for each ATM strike, buy next OTM strike up to price cap
        - if funds remaining, repeat ATM->OTM ladder
        - if funds remaining, but can't afford ATM anymore, keep trying
          the OTM ladder to spend remaining funds until exhausted.
    """

    symbol: str = field(init=False)
    direction: str = field(init=False)
    amount: float = field(init=False)
    gaps: int = field(init=False)
    atm: int = field(init=False)
    expirationAway: int = field(init=False)
    percentageRun: float = field(init=False)
    preview: list[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "symbol",
                convert=lambda x: x.upper(),
                verify=lambda x: " " not in x,
                desc="Underlying for contracts",
            ),
            DArg(
                "direction",
                convert=lambda x: x.upper()[0],
                verify=lambda x: x in {"P", "C"},
                desc="side to buy / trade kind to execute (puts or calls)",
            ),
            DArg("amount", convert=float, desc="Maximum dollar amount to spend"),
            DArg(
                "gaps",
                convert=int,
                desc="pick strikes N apart (e.g. use 2 for room to lock gains with butterflies)",
            ),
            DArg(
                "atm",
                convert=int,
                desc="Number of strikes away from ATM to target for starting. Can be negative to start ITM.",
            ),
            # TODO: add feature where we could just specify "OPEX" for next opex? Just need to calculate the next 3rd fridayd date.
            DArg(
                "expirationAway",
                convert=int,
                desc="use N next expiration date (0 is NEAREST, 1 is NEXT, 2 is NEXT NEXT, ...)",
            ),
            DArg(
                "percentageRun",
                convert=float,
                # don't go negative or we infinite loop!
                # TODO: allow INNER (negative) percentages/offsets so we back up with ITM strikes instead of only OTM strikes...
                verify=lambda x: x >= 0,
                desc="percentage from current price for strikes to buy (0.03 => buy up to 3% OTM strikes, etc) — OR — if 1+ then SKIP N STRIKES FROM ATM for first buy",
            ),
            DArg(
                "*preview",
                desc="If any other arguments present, order not placed, only order logic reported.",
            ),
        ]

    async def run(self):
        # Fetch data for our calculations:
        #   - ATR with moving 3 day window
        #   - ATR with moving 20 day window
        #   - current strikes / dates
        #   - live quote for underlying

        # TODO: may need symbol replacement for SPXW -> SPX etc?
        # TODO: make atr fetch optional and make chains fetch against
        #       an external API if configured.
        # TODO: this whole atr fetching / calculation / partitioning strikes
        #       should be an external reusable function (because it's complicating
        #       the logic flow of "calculate strike widths, place order"
        if True:
            # skip ATR for now
            initSym = self.symbol

            # see if we're using an alias and, if so, resolve it and attach it locally
            if initSym.startswith(":"):
                found, contract = self.state.quoteResolve(self.symbol)
                if not found:
                    logger.error("[{}] Symbol not found for mapping?", initSym)
                    return None

                assert found
                self.symbol = found

            # async run all URL fetches and data updates at once
            strikes = await self.runoplive(
                "chains",
                self.symbol,
            )

            if not strikes:
                logger.error("[{}] No strikes found?", self.symbol)
                return None

            initSym = self.symbol
            if initSym.upper() == "SPXW":
                initSym = "SPX"

            strikes = strikes[initSym]

            # also make sure quote for the underlying is populated...
            await self.runoplive(
                "add",
                f'"{initSym}"',
            )
        else:
            atrReqFast = dict(cmd="atr", sym=self.symbol, avg=3, back=1)
            atrReqSlow = dict(cmd="atr", sym=self.symbol, avg=20, back=1)
            async with aiohttp.ClientSession() as cs:
                # This URL is a custom historical reporting endpoint providing various
                # technical stats on symbols given arbitrary parameters, but also requires
                # you have a DB of potentially years of daily bars for each stock.
                urls = [
                    cs.get("http://127.0.0.1:6555/", params=p)
                    for p in [atrReqFast, atrReqSlow]
                ]

                # request chains and live quote data using the language command syntax
                dataUpdates = [
                    self.runoplive(
                        "chains",
                        self.symbol,
                    ),
                    self.runoplive(
                        "add",
                        f'"{self.symbol}"',
                    ),
                ]

                # async run all URL fetches and data updates at once
                mfast_, mslow_, strikes, _ = await asyncio.gather(*(urls + dataUpdates))
                strikes = strikes[self.symbol]

                # async resolve response bodies through the JSON parser
                mfast, mslow = await asyncio.gather(
                    *[m.json() for m in [mfast_, mslow_]]
                )

        # logger.info("Got MFast, MSlow: {}", [mfast, mslow])

        # Get a valid expiration from our strikes...
        # Note: we check if it matches below, then back up if doesn't
        # TODO: allow selecting future expirations too
        # Note: this is *not* the target options expiration date, but we use the
        #       current date to bisect all chains to discover the nearest date to now.
        now = whenever.ZonedDateTime.now("US/Eastern")

        # if after market close, use next day
        if (now.hour, now.minute) >= (16, 15):
            now = now.add(days=1, disambiguate="compatible")

        expTry = now.date()
        # Note: Expiration formats in the dict are *full* YYYYMMDD, not OCC YYMMDD.
        expirationFmt = f"{expTry.year}{expTry.month:02}{expTry.day:02}"

        # get nearest expiration from today (note: COULD BE TODAY!)
        # if today is NOT an expiration, this picks the CLOSEST expiration date.
        # TODO: allow selecting future expirations.
        sstrikes = sorted(strikes.keys())

        # why doesn't this work for VIX?
        # logger.info("Strikes are: {}", sstrikes)

        useExpIdx = bisect.bisect_left(sstrikes, expirationFmt)

        # this is a bit weird, but we want to find the NEXT NEAREST expiration date (even if today),
        # which is the 'useExpIdx', then we want the requested expire away, which is 'self.expirationAway'
        # distance from the currently found expiration date, so we get a slice of the expirationAway length,
        # then we take the last element which is our actually requested expiration away date.
        if not sstrikes:
            logger.error(
                "[{}] No strikes found? Does this symbol have options?", self.symbol
            )
            return None

        useExp = sstrikes[useExpIdx : useExpIdx + 1 + self.expirationAway][-1]

        assert useExp in strikes

        useChain = strikes[useExp]

        logger.info(
            "[{} :: {} :: {}] Using expiration {} having chain: {}",
            initSym,
            self.direction,
            useExpIdx,
            useExp,
            useChain,
        )

        if False:
            maximumATR: float = 0
            prevClose: float = 0
            prevHigh: float = 0

            # just not running for now, skipping ATR limits
            for df in (mfast, mslow):
                data = df["data"]
                date: str = list(data.keys())[-1]
                fordate = data[date]
                atr: float = fordate["atr"]

                if atr > maximumATR:
                    maximumATR = atr
                    prevClose = fordate["close"]
                    prevHigh = fordate["high"]

        # quote is already live in quote state...

        # HACK TO GET AROUND NOT HAVING INDEX QUOTES
        useQuoteSymbol = self.symbol
        if useQuoteSymbol in {"SPX", "SPXW"}:
            # only works if you're subscribing to Index(SPX, CBOE) things. can also replace with "ES"
            # if you want to base of the current /ES quote...
            # TODO: should we do something like "overnight and pre-market, use ES, else use live SPX?"
            useQuoteSymbol = "SPX"
        elif useQuoteSymbol in {"NDX", "NDXW"}:
            # Same problem with SPX, but NDX requires different data subscriptions, but
            # you may have /MNQ or /NQ instead which may be ~okay?
            useQuoteSymbol = "MNQ"

        quote = self.state.quoteState[useQuoteSymbol]

        # if we subscribed to new quotes, wait a little while for
        # the IBKR streaming API to start populating our quotes...
        for i in range(0, 5):
            # currentLow: float = quote.last # RETURN TO: quote.low
            # TEMPORARY HACK BECAUSE OVER WEEKEND "SPX" returns NAN for LAST and LOW. SIGH.
            currentLow: float = quote.last if quote.last == quote.last else quote.close

            # note: this can break during non-market hours when
            #       IBKR returns random bad values for current
            #       prices (also after market hours their bid/ask
            #       no longer populates so we can't just make it
            #       a midpoint).
            # Though, quote.last seems more reliable than quote.marketPrice()
            # for any testing after hours/weekends/etc.
            # TODO: could also use 'underlying' value from full OCC quote, but need
            #       a baseline to get the first options quotes anyway...

            # used to look up the ATM amount here:
            currentPrice: float = (
                quote.last if quote.last == quote.last else quote.close
            )

            if any(np.isnan(np.array([currentLow, currentPrice]))):
                logger.warning(
                    "[{} :: {}] [{}] Quotes not all populated ({}). Waiting for activity...",
                    self.symbol,
                    self.direction,
                    i,
                    quote.last,
                )

                await asyncio.sleep(0.140)
            else:
                logger.info(
                    "[{} :: {}] Using quote last price: {}",
                    self.symbol,
                    self.direction,
                    currentPrice,
                )
                break

        # sort a nan-free list of atr goals
        # (note: sorting here because we want to print it in the log, so
        #        below instead of max() we can also just do [-1] since we
        #        already sorted here)
        # TODO: expose this sorted list to the user so they can pick their
        #       "upper aggressive level" tolerance (low, medium, high)
        usingCalls = self.direction == "C"

        if currentPrice != currentPrice:
            logger.error("No current price found, can't continue.")
            return None

        # boundaryPrice hack because we don't have the ATR server at the moment
        # NOTE: this gets updated for range extremes during the strike calculation loop.
        boundaryPrice = currentPrice

        if False:
            # not using ATR for now
            if usingCalls:
                # call goals sort forward with the highest range as the rightmost value
                atrGoals = [
                    x
                    for x in sorted(
                        [
                            prevClose + maximumATR,
                            prevHigh + maximumATR,
                            currentLow + maximumATR,
                        ]
                    )
                    if x == x  # drop nan values if we are missing a quote
                ]
            else:
                # put goals sort backwards with lowest range as the rightmost value
                atrGoals = [
                    x
                    for x in reversed(
                        sorted(
                            [
                                prevClose - maximumATR,
                                prevHigh - maximumATR,
                                currentLow - maximumATR,
                            ]
                        )
                    )
                    if x == x  # drop nan values if we are missing a quote
                ]

        # boundary value (the up/down we don't want to cross)
        # is always last element of the sorted goals list
        if False:
            boundaryPrice = atrGoals[-1]

        # it's possible current-day performance is better (or worse) than
        # expected ATR-calculated expected performance, so if underlying
        # is already beyond the maximum ATR calculation
        # (either above it for calls or under it for puts), then we
        # need to adjust the full boundary expectation off the current
        # price because maybe this is going to be a double ATR day.
        # (otherwise, if we don't adjust, the conditions below end
        #  up empty because it tries to interrogate an empty range
        #  because currentPrice would be higher than expected prices
        #  to buy (for calls), so we'd never generate a buyQty)
        if False:
            if usingCalls:
                if boundaryPrice <= currentPrice:
                    boundaryPrice += maximumATR
            else:
                if boundaryPrice >= currentPrice:
                    boundaryPrice -= maximumATR

        # collect strikes near the current price in gaps as requested
        # until we reach the maximum price
        # (and walk _backwards_ if doing puts!)
        firstStrikeIdx = find_nearest(useChain, currentPrice)

        buyStrikes = []

        # if puts, walk towards lower strikes instead of higher strikes.
        directionMul = 1 if usingCalls else -1

        # move strike by 'atm' positions in the direction of the request (LOWER PRICE for calls, HIGHER PRICE for puts)
        firstStrikeIdx += directionMul * self.atm

        # If asking for negative gaps (more ITM instead of more OTM) then our
        # direction is inverted from normal for the call/put strike discovery.
        # (e.g. gaps >= 0 == go MORE OTM; gaps < 0 == go MORE ITM)
        # TODO: fix direction for -0 (need go use ±1 instead of 0 for no gaps...)
        # directionMul = -directionMul if abs(self.gaps) != self.gaps else directionMul

        # the range step paraemter is the "+=" increment between
        # iterations, so step=2 returns 0, 2, 4, ...;
        # step=3 returns 0, 3, 6, 9, ...
        # but our "gaps" params is numbers BETWEEN steps, so we
        # need steps=(gaps+1) becaues the step is 'inclusive' of the
        # next result value, but our 'gaps' is fully exclusive gaps.

        # user-defined percentage gap for range buys...

        if self.percentageRun == 0:
            # If NO WIDTH specified, use current ATM pick (potentially moved by 'gaps' requested to start)
            poffset = self.gaps
            picked = useChain[firstStrikeIdx]
            logger.info("No width requested, so using: {}", picked)
            buyStrikes.append(picked)
        elif self.percentageRun >= 1:
            poffset = int(self.percentageRun)
            try:
                picked = useChain[
                    firstStrikeIdx + (poffset * (1 if usingCalls else -1))
                ]
            except:
                # lazy catch-all in case 'poffset' overflows the array extent
                picked = useChain[firstStrikeIdx]

            logger.info("Requested ATM+{} width, so using: {}", poffset, picked)
            buyStrikes.append(picked)
        else:
            pr = 1 + self.percentageRun
            while len(buyStrikes) < 1:
                # if we didn't find a boundary price, then extend it a bit more.
                # NOTE: This is still a hack because our ATR server isn't live again so we're guessing
                #       a 0.5% range?
                # TODO: allow fast buy with delta calculation (0.5 -> 0.25 etc)
                boundaryPrice = boundaryPrice * pr if usingCalls else boundaryPrice / pr

                # TODO: change initial position back to 0 from len(buyStrikes) when we restore
                #       proper ATR API reading.
                # TODO: fix gaps calculation when doing negative gaps
                # TODO: maybe make gaps=1 be NEXT strike so gaps=-1 is PREV strike for going more ITM?
                for i in range(
                    len(buyStrikes), 100 * directionMul, (self.gaps + 1) * directionMul
                ):
                    idx = firstStrikeIdx + i

                    # if we walk over or under the strikes, we are done.
                    if idx < 0 or idx >= len(useChain):
                        break

                    strike = useChain[idx]

                    logger.info(
                        "Checking strike vs. boundary: ${:,.3f} v ${:,.3f}",
                        strike,
                        boundaryPrice,
                    )

                    # only place orders up to our maximum theoretical price
                    # (either a static percent offset from current OR the actual High-to-Low ATR)
                    if usingCalls:
                        # calls have an UPPER CAP
                        if strike > boundaryPrice:
                            break
                    else:
                        # puts have a LOWER CAP
                        if strike < boundaryPrice:
                            break

                    buyStrikes.append(strike)

        logger.info(
            "[{} :: {}] Selected strikes to purchase: {}",
            self.symbol,
            self.direction,
            buyStrikes,
        )

        STRIKES_COUNT = len(buyStrikes)

        # get quotes for strikes...
        occs = [
            f"{self.symbol}{useExp[2:]}{self.direction[0]}{int(strike * 1000):08}"
            for strike in buyStrikes
        ]

        logger.info("Adding quotes for: {}", " ".join(occs))
        await self.runoplive(
            "add",
            " ".join(occs),
        )

        buyQty: dict[str, int] = defaultdict(int)

        remaining = self.amount
        spend = 0
        skip = 0
        while remaining > 0 and skip < STRIKES_COUNT:
            logger.info(
                "[{} :: {}] Remaining: ${:,.2f} plan {}",
                self.symbol,
                self.direction,
                remaining,
                " ".join(f"{x}={y}" for x, y in buyQty.items()) or "[none yet]",
            )
            for idx, (strike, occ) in enumerate(
                zip(
                    buyStrikes,
                    occs,
                )
            ):
                # TODO: for VIX, expire date is N, but contract date is N+1, so we need
                #       to do calendar math for "add 1 day" to VIX symbols...

                # (was this fixed by adding dual symbols to the Option constructor?)
                occForQuote = occ

                # multipler is a string of a number because of course it is.
                try:
                    qs = self.state.quoteState[occForQuote]
                except:
                    logger.error(
                        "[{}] Quote doesn't exist? Can't continue!", occForQuote
                    )
                    return None

                multiplier = float(qs.contract.multiplier or 1)

                ask = self.state.quoteState[occForQuote].ask * multiplier

                logger.info("Iterating [cost ${:,.2f}]: {}", ask, occForQuote)

                # if quote not populated, wait for it...
                try:
                    for i in range(0, 25):
                        if ask is not None:
                            break

                        # else, ask is not populated, so try to wait for it...
                        logger.warning("Ask not populated. Waiting...")

                        await asyncio.sleep(0.075)

                        ask = qs.ask * multiplier
                    else:
                        # ask failed to populate, so use MP
                        logger.warning(
                            "Failed to populate quote, so using market price..."
                        )
                        ask = qs.marketPrice() * multiplier
                except KeyboardInterrupt:
                    logger.error(
                        "[FAST Order] Request termination received. Abandoning! No orders were placed."
                    )
                    return  # Control-C pressed. Goodbye.

                # attempt to weight lower strikes for optimal
                # delta capture while still getting gamma exposure
                # a little further out.
                # Current setting is:
                #   - first strike (closest to ATM) tries 3->2->1->0 buys
                #   - second strike (next furthest OTM) tries 2->1->0 buys
                #   - others try 1 buy per iteration
                for i in reversed(range(1, 4 if idx == 0 else 3 if idx == 1 else 2)):
                    if False:
                        logger.debug(
                            "[{}] Checking ({} * {}) + {} < {}",
                            occForQuote,
                            ask,
                            i,
                            spend,
                            self.amount,
                        )

                    if (ask * i) + spend < self.amount:
                        buyQty[occ] += i

                        spend += ask * i
                        remaining -= ask * i
                        break
                else:
                    # since we are only reducing 'remaining' on spend, we need
                    # another guaranteed terminiation condtion, so now we pick
                    # "terminate if skip is larger than the strikes we are buying"
                    skip += 1
                    # logger.debug(
                    #    "[{}] Skipping because doesn't fit remaining spend...", occ
                    # )

        logger.info(
            "[{} :: {}] Buying plan for {} contracts (est ${:,.2f}): {}",
            self.symbol,
            self.direction,
            sum(buyQty.values()),
            spend,
            " ".join(f"{x}={y}" for x, y in buyQty.items()),
        )

        if False:
            logger.info(
                "[{} :: {}] ATR ({:,.2f}) Estimates: {}",
                self.symbol,
                self.direction,
                maximumATR,
                atrGoals,
            )
            logger.info(
                "[{} :: {}] Chosen ATR: {} :: {}",
                self.symbol,
                self.direction,
                maximumATR,
                boundaryPrice,
            )

        logger.info(
            "{}[{} :: {}] Ordering ${:,.2f} of {} apart between {:,.2f} and {:,.2f} ({:,.2f})",
            "[PREVIEW] " if self.preview else "",
            self.symbol,
            self.direction,
            self.amount,
            self.gaps,
            currentPrice,
            boundaryPrice,
            abs(currentPrice - boundaryPrice),
        )

        # now use buyDict to place orders at QTY and live track them until
        # hitting the ask.

        # don't continue to order placement if this is a preview-only request
        if self.preview:
            return None

        # qualify contracts for ordering
        # don't qualify here because they are qualified in the async order runners instead
        # contracts = {occ: contractForName(occ) for occ in buyQty.keys()}
        # await self.state.qualify(*contracts.values())

        # assemble coroutines with parameters per order
        placement = []
        for occ, qty in buyQty.items():
            # TODO: FIX HACK NAME CRAP
            qs = self.state.quoteState[occ.replace("SPX", "SPXW")]
            limit = round((qs.bid + qs.ask) / 2, 2)

            # if for some reason bid/ask aren't populated, wait for them...
            while np.isnan(limit):
                await asyncio.sleep(0.075)
                limit = round((qs.bid + qs.ask) / 2, 2)

            # buy order algo hackery...
            algo = "AF"
            if "SPXW" in occ:
                algo = "PRTMKT"

            placement.append((occ, qty, limit, algo))
            # placement.append(
            #     self.state.placeOrderForContract(
            #         occ, True, contracts[occ], qty, limit, "LMT + ADAPTIVE + FAST"
            #     )
            # )

        # launch all orders concurrently
        # placed = await asyncio.gather(*placement)

        # TODO: create follow-the-spread refactor so we can
        # live follow all new strike order simmultaenously.
        # (currently around line 585)

        # re-use the "BUY" self-adjusting price algo here to place orders
        # for all our calculated strikes.
        placed = await asyncio.gather(
            *[
                self.runoplive("buy", f"{occ} {qty} {algo}")
                for occ, qty, _limit, algo in placement
            ]
        )

        logger.info("Placed results: {}", pp.pformat(placed))

        return True


@dataclass
class IOpOrderLimit(IOp):
    """Create a new buy or sell order using limit, market, or algo orders."""

    args: list[str] = field(init=False)

    def argmap(self):
        # allow symbol on command line, optionally
        return [DArg("*args")]

    async def run(self):
        promptSide = [
            Q(
                "Side",
                choices=[
                    "Buy to Open",
                    "Sell to Close",
                    "Sell to Open",
                    "Buy to Close",
                    "Butterfly Lock Gains to Close",
                ],
            ),
        ]

        gotside = await self.state.qask(promptSide)

        try:
            assert gotside
            side = gotside["Side"]
            isClose = "to Close" in side
            isSell = "Sell" in side
            isButterflyClose = "Butterfly" in side
        except:
            # user canceled prompt, so skip
            return None

        if isClose:
            port = self.ib.portfolio()
            # Choices have a custom format/title string, but the
            # return value is the PortfolioItem object which has:
            #   .position for the entire quantity in the account
            #   .contract for the contract object to use with orders

            if isSell:
                # if SELL, show only LONG positions
                # TODO: exclude current orders already fully booked!
                portchoices = [
                    Choice(strFromPositionRow(p), p)
                    for p in sorted(port, key=portSort)
                    if p.position > 0
                ]
            else:
                # if BUY, show only SHORT positions
                portchoices = [
                    Choice(strFromPositionRow(p), p)
                    for p in sorted(port, key=portSort)
                    if p.position < 0
                ]

            promptPosition = [
                Q("Symbol", choices=portchoices),
                Q("Price"),
                Q("Quantity"),
                ORDER_TYPE_Q,
            ]
        else:
            # OPENING
            # provide any arguments as pre-populated symbol by default
            promptPosition = [
                Q(
                    "Symbol",
                    value=" ".join(
                        [
                            (await self.state.positionalQuoteRepopulate(x))[0]
                            or "[NOT FOUND]"
                            for x in self.args
                        ]
                    ),
                ),
                Q("Price"),
                Q("Quantity"),
                ORDER_TYPE_Q,
            ]

        got = await self.state.qask(promptPosition)

        if not got:
            # user canceled at position request
            return None

        # if user canceled the form, just prompt again
        try:
            # sym here is EITHER:
            #   - input string provided by user
            #   - the cached PortfolioItem if this is a Close order on current position
            symInput = got["Symbol"]

            contract: Contract | None = None
            portItem: PortfolioItem | None = None

            # if sym is a CONTRACT, make it explicit and also
            # retain the symbol name independently
            if isinstance(symInput, PortfolioItem):
                portItem = symInput
                contract = portItem.contract
                sym = symInput.contract.symbol
            else:
                # else, is just a string
                sym = symInput

            qty = got["Quantity"]
            if qty:
                # only convert non-empty strings to floats
                qty = float(got["Quantity"])
            elif isClose:
                # else, we have an empty string so assume we want to use ALL position
                logger.warning("No quantity provided. Using ALL POSITION as quantity!")
                qty = None
            else:
                logger.error(
                    "No quantity provided and this isn't a closing order, so we can't order anything!"
                )
                return

            price = float(got["Price"])
            isLong = gotside["Side"].startswith("Buy")
            orderType = got["Order Type"]
        except:
            # logger.exception("Failed to get field?")
            logger.exception("Order creation canceled by user")
            return None

        # if order is To Close, then find symbol inside our active portfolio
        if isClose:
            # abs() because for orders, positions are always positive quantities
            # even though they are reported as negative shares/contracts in the
            # order status table.
            assert portItem

            isShort = portItem.position < 0
            qty = abs(portItem.position) if (qty is None or qty == -1) else qty

            if contract is None:
                logger.error("Symbol [{}] not found in portfolio for closing!", sym)

            if isButterflyClose:
                strikesDict = await self.runoplive(
                    "chains",
                    sym,
                )
                strikesUse = strikesDict[sym]

                # strikes are in a dict by expiration date,
                # so symbol AAPL211112C00150000 will have expiration
                # 211112 with key 20211112 in the strikesDict return
                # value from the "chains" operation.
                # Note: not year 2100 compliant.
                strikesFound = strikesUse["20" + sym[-15 : -15 + 6]]

                currentStrike = float(sym[-8:]) / 1000
                pos = find_nearest(strikesFound, currentStrike)
                # TODO: filter this better if we are at top of chain
                (l2, l1, current, h1, h2) = strikesFound[pos - 2 : pos + 3]

                # verify we found the correct midpoint or else the next strike
                # calculations will be all bad
                assert (
                    current == currentStrike
                ), f"Didn't find strike in chain? {current} != {currentStrike}"

                underlying = sym[:-15]
                contractDateSide = sym[-15 : -15 + 7]

                # closing a short with butterfly is a middle BUY
                # closing a long with butterfly is a middle SELL
                if isShort:
                    # we are short a high strike, so close on lower strikes.
                    # e.g. SHORT -1 $150p, so BUY 2 $145p, SELL 1 140p
                    ratio2 = f"{underlying}{contractDateSide}{int(l1 * 1000):08}"
                    ratio1 = f"{underlying}{contractDateSide}{int(l2 * 1000):08}"
                    bOrder = f"buy 2 {ratio2} sell 1 {ratio1}"
                else:
                    # we are long a low strike, so close on higher strikes.
                    # e.g. LONG +1 $150c, so SELL 2 $155c, BUY 1 $160c
                    ratio2 = f"{underlying}{contractDateSide}{int(h1 * 1000):08}"
                    ratio1 = f"{underlying}{contractDateSide}{int(h2 * 1000):08}"
                    bOrder = f"sell 2 {ratio2} buy 1 {ratio1}"

                # use 'bOrder' for 'sym' string reporting and logging going forward
                sym = bOrder
                logger.info("[{}] Requesting butterfly order: {}", sym, bOrder)
                orderReq = self.state.ol.parse(bOrder)
                contract = await self.state.bagForSpread(orderReq)
        else:
            contract = contractForName(sym)

        if contract is None:
            logger.error("Not submitting order because contract can't be formatted!")
            return None

        if not isButterflyClose:
            # butterfly spread is already qualified when created above
            (contract,) = await self.state.qualify(contract)

        # LIMIT CMD
        return await self.state.placeOrderForContract(
            sym,
            isLong,
            contract,
            PriceOrQuantity(qty, is_quantity=True),
            Decimal(str(price)),
            orderType,
        )

        # TODO: allow opt-in to midpoint price adjustment following.


@dataclass
class IOpOrderCancel(IOp):
    """Cancel waiting orders via order ids or interactive prompt."""

    orderids: list[int] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*orderids",
                lambda xs: set([int(x) if x.isdigit() else x for x in xs]),
                errmsg="Order IDs can be order id integers or glob strings for symbols to cancel.",
            )
        ]

    async def run(self):
        # If order ids not provided via command, show a selectable list of all orders
        if not self.orderids:
            ords = [
                CB(
                    "Orders to Cancel",
                    choices=[
                        Choice(
                            f"[{o.order.orderId}] {o.order.action} {self.state.nameForContract(o.contract)} {o.order.orderType:>5} ({o.order.totalQuantity} * ${o.order.lmtPrice:.2f}) == ${float(o.order.totalQuantity) * float(o.order.lmtPrice) * self.state.multiplier(o.contract):,.{self.state.decimals(o.contract) or 4}f} status:{o.log[-1].status} events:{len(o.log)}",  # type: ignore
                            o.order,
                        )
                        for o in sorted(
                            filter(
                                lambda x: x.orderStatus.clientId == self.state.clientId
                                and x.log[-1] != "Inactive",
                                self.ib.openTrades(),
                            ),
                            key=tradeOrderCmp,
                        )
                    ],
                )
            ]
            oooos = await self.state.qask(ords)

            if not oooos:
                logger.info("Cancel canceled by user cancelling")
                return

            oooo = oooos["Orders to Cancel"]
        else:
            # else, use order IDs given on command line to find existing orders to cancel
            oooo = []

            # if orderid is special value "safe" then only cancel orders WITHOUT parent IDs (i.e. don't cancel active child orders).
            safeRemove = "safe" in self.orderids

            for orderid in self.orderids:
                # These aren't indexed in anyway, so just n^2 search, but the
                # count is likely to be tiny overall.
                # TODO: we could maintain a cache of active orders indexed by orderId
                for t in self.ib.openTrades():
                    # we can't cancel orders not on our current orderId (clientId==0 can see all orders, but it can't modify them)
                    if t.orderStatus.clientId != self.state.clientId:
                        continue

                    # if provided direct order id integer, just check directly...
                    if isinstance(orderid, int):
                        if t.order.orderId == orderid:
                            oooo.append(t.order)
                    else:
                        # else, is either a symbol name or glob to try...
                        name = t.contract.localSymbol.replace(" ", "")

                        # also allow evict :N if we just have a quote for it handy...
                        # TODO: actually move this out to the arg pre-processing step so we don't run it each time
                        if orderid[0] == ":":
                            orderid, _contract = self.state.quoteResolve(orderid)

                        if fnmatch.fnmatch(name, orderid):
                            # if any order id is 'safe' then then only cancel orders WITHOUT parent IDs.
                            # (i.e. don't cancel active child orders, only cancel parent orders or full orders without children)
                            if safeRemove and not t.order.parentId:
                                oooo.append(t.order)
                            elif not (
                                t.order.parentId and t.log[-1].status == "PreSubmitted"
                            ):
                                # Don't manually cancel orders having parent ids if not submitted when using wildcards
                                # (because removing the parent order itself will cancel the attached orders)
                                oooo.append(t.order)

        if not oooo:
            logger.error("[{}] No match for orders cancel!", self.orderids)
            return

        for n, o in enumerate(oooo, start=1):
            logger.info("[{} of {}] Matched order to cancel: {}", n, len(oooo), o)
            self.ib.cancelOrder(o)


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


@dataclass
class IOpBalance(IOp):
    """Return the currently cached account balance summary."""

    fields: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*fields")]

    async def run(self):
        ords = self.state.summary

        # if specific fields requested, compare by case insensitive prefix
        # then only output matching fields
        if self.fields:
            send = {}
            for k, v in ords.items():
                for field in self.fields:
                    if k.lower().startswith(field.lower()):
                        send[k] = v
            logger.info("{}", pp.pformat(send))
        else:
            # else, no individual fields requested, so output the entire summary
            logger.info("{}", pp.pformat(ords))


@dataclass
class IOpPositions(IOp):
    """Print datatable of all positions."""

    symbols: set[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: set([sym.upper() for sym in x]))]

    def totalFrame(self, df, costPrice=False):
        if df.empty:
            return None

        # Add new Total index row as column sum (axis=0 is column sum; axis=1 is row sum)
        totalCols = [
            "position",
            "marketValue",
            "totalCost",
            "unrealizedPNLPer",
            "unrealizedPNL",
            "dailyPNL",
            "%",
        ]

        # For spreads, we want to sum the costs/prices since they
        # are under the same order (hopefully).
        if costPrice:
            totalCols.extend(["averageCost", "marketPrice"])

        df.loc["Total"] = df[totalCols].sum(axis=0)
        t = df.loc["Total"]

        # don't do total maths unless totals actually exist
        # (e.g. if you have no open positions, this math obviously can't work)
        if t.all():
            df.loc["Total", "%"] = (
                (t.marketValue - t.totalCost) / ((t.marketValue + t.totalCost) / 2)
            ) * 100

        # Calculated weighted percentage ownership profit/loss...
        try:
            df["w%"] = df["%"] * (abs(df.totalCost) / df.loc["Total", "totalCost"])
            df.loc["Total", "w%"] = df["w%"].sum()
        except:
            # you probably don't have any positions...
            df["w%"] = 0
            df.loc["Total", "w%"] = 0

        # give actual price columns more detail for sub-penny prices
        # but give aggregate columns just two decimal precision
        detailCols = [
            "marketPrice",
            "averageCost",
            "strike",
        ]
        simpleCols = [
            "%",
            "w%",
            "unrealizedPNLPer",
            "unrealizedPNL",
            "dailyPNL",
            "totalCost",
            "marketValue",
        ]

        # convert columns to all strings because we format them as nice to read money strings, but pandas
        # doesn't like replacing strings over numeric-typed columns anymore.
        df[simpleCols] = df[simpleCols].astype(str)
        df[detailCols] = df[detailCols].astype(str)
        df["conId"] = df["conId"].map(lambda x: f"{int(x) if x and x == x else ''}")

        df.loc[:, detailCols] = (
            df[detailCols]
            .astype(float, errors="ignore")
            .map(
                lambda x: fmtPrice(float(x))
                if (x and (isinstance(x, str) and " " not in x))
                else x
            )
        )
        df.loc[:, simpleCols] = df[simpleCols].astype(float).map(lambda x: f"{x:,.2f}")

        # show fractional shares only if they exist
        defaultG = ["position"]
        df[defaultG] = df[defaultG].astype(str)
        df.loc[:, defaultG] = df[defaultG].astype(float).map(lambda x: f"{x:,.10g}")

        # manually override the string-printed 'nan' from .map() of totalCols
        # for columns we don't want summations of.
        # df.at["Total", "closeOrder"] = ""

        # if not costPrice:
        #     df.at["Total", "marketPrice"] = ""
        #     df.at["Total", "averageCost"] = ""

        return df

    async def run(self):
        ords = self.ib.portfolio()
        # logger.info("port: {}", pp.pformat(ords))

        backQuickRef = []
        populate = []

        for o in ords:
            # Store similar details together for same-symbol spread discovery
            backQuickRef.append((o.contract.secType, o.contract.symbol, o.contract))

            # fetch qualified contract from not completely populated port contract
            # (there's no speed difference fetching them individually vs collectively up front,
            #  so it's simpler just to run them one-by-one here as we iterate the portfolio)
            (contract,) = await self.state.qualify(o.contract)

            # fetch qualified contract metadata so we can use proper decimal rounding for display
            # (otherwise, we get IBKR default 8 digit floats everywhere)
            digits = self.state.decimals(contract)

            # Nice debug output showing full contracts.
            # TODO: enable global debug flags for showing these
            # maybe just enable logger.debug mode via a command?
            # logger.info("{}", o.contract)

            make: dict[str, Any] = {}

            # 't' used for switching on OPT/WAR/STK/FUT types later too.
            t = o.contract.secType

            make["conId"] = o.contract.conId
            make["type"] = t
            make["sym"] = o.contract.symbol

            # allow user input to compare against any of the actual symbols representing the instrument
            checkSymbols = {
                o.contract.symbol,
                o.contract.localSymbol,
                o.contract.localSymbol.replace(" ", ""),
                o.contract.tradingClass,
            }

            # TODO: update this to allow glob matching wtih fnmatch.filter(sourceCollection, targetGlob)
            if self.symbols and not (checkSymbols & self.symbols):
                continue

            # logger.info("contract is: {}", o.contract)
            if isinstance(o.contract, (Option, Warrant, FuturesOption)):
                try:
                    make["date"] = dateutil.parser.parse(
                        o.contract.lastTradeDateOrContractMonth
                    ).date()
                except:
                    logger.error("Row didn't have a good date? {}", o)
                    pass

                make["strike"] = o.contract.strike
                make["PC"] = o.contract.right

            make["exch"] = o.contract.primaryExchange[:3]
            make["position"] = o.position

            make["marketPrice"] = round(o.marketPrice, digits + 1)

            close = self.state.orderPriceForContract(o.contract, o.position)

            # if it's a list of tuples, break them by newlines for display
            multiplier = float(o.contract.multiplier or 1)
            isLong = o.position > 0

            # TODO: fix this logic because technically if the close order isn't the entire size
            #       of the current position, we should always report in the list format to show pairs of size+price
            #       since we are only disposing of a sub-quantity of the entire position
            if isinstance(close, list):
                closingSide = " ".join([str(x) for x in close])
                make["closeOrderValue"] = " ".join(
                    [f"{size * price * multiplier:,.2f}" for size, price in close]
                )
                make["closeOrderProfit"] = " ".join(
                    [
                        f"{-((size * price * multiplier) + (o.averageCost * size)):,.2f}"
                        for size, price in close
                    ]
                    if isLong
                    else [
                        f"{(size * o.averageCost) - (size * price * multiplier):,.2f}"
                        for size, price in close
                    ]
                )
            else:
                closingSide = close
                # logger.info("FIELDS HERE: {}", pp.pformat(dict(close=close, pos=o.position, mul=multiplier, avg=o.averageCost, o=o)))
                make["closeOrderValue"] = f"{close * o.position * multiplier:,.2f}"

                # We use addition for the Profit here because the 'close' price is negaive for sales which makes the math work out.
                make["closeOrderProfit"] = (
                    # Longs have NEGATIVE CLOSING PRICE and POSIIVE COST,
                    # so we ADD THE EXIT PRICE to the COST BASIS (-exit + +cost) which is still a negative (credit)
                    # for a profit, but we invert it to show positive for profit here since this is a P&L column.
                    f"{-((close * o.position * multiplier) + (o.averageCost * o.position)):,.2f}"
                    if isLong
                    # Shorts have POSITIVE CLOSING PRICE and NEGATIVE COST
                    # TODO: verify short math works correct where profit is positive and loss is negative
                    else f"{(o.averageCost * o.position) - (close * o.position * multiplier):,.2f}"
                )

            make["closeOrder"] = closingSide
            make["marketValue"] = o.marketValue
            make["totalCost"] = o.averageCost * o.position
            make["unrealizedPNLPer"] = o.unrealizedPNL / abs(o.position)
            make["unrealizedPNL"] = o.unrealizedPNL

            try:
                # Note: dailyPnL per-position is only subscribed on the client where the order
                #       originated, so you may get 'dailyPnL' position errors if you view
                #       positions on a different client than the original.
                make["dailyPNL"] = self.state.pnlSingle[o.contract.conId].dailyPnL

                # API issue where it returns the largest value possible if not populated.
                # same as: sys.float_info.max:
                if not isset(make["dailyPNL"]):
                    make["dailyPNL"] = -1
            except:
                logger.warning(
                    "Subscribing to live PNL updates for: {}",
                    o.contract.localSymbol or o.contract.symbol,
                )

                # if we didn't have a PnL, attempt to subscribe it now anyway...
                # (We can have an unsubscribed PnL if we have live positions created today
                #  on another client or we have some positions just "show up" like getting assigned
                #  long shares from short puts or getting assigned short shares from short calls)
                self.state.pnlSingle[o.contract.conId] = self.ib.reqPnLSingle(
                    self.state.accountId, "", o.contract.conId
                )

                pass

            multiplier = float(o.contract.multiplier or 1)
            if t == "FUT":
                make["averageCost"] = round(o.averageCost / multiplier, digits + 1)
                make["%"] = (
                    (o.marketPrice * multiplier - o.averageCost) / o.averageCost * 100
                )
            elif t == "BAG":
                logger.info("available: {}", o)
            elif t in {"OPT", "FOP"}:
                # For some reason, IBKR reports marketPrice
                # as the contract price, but averageCost as
                # the total cost per contract. shrug.
                make["%"] = (
                    (o.marketPrice * multiplier - o.averageCost) / o.averageCost * 100
                )

                # show average cost per share instead of per contract
                # because the "marketPrice" live quote is the quote
                # per share, not per contract.
                make["averageCost"] = round(o.averageCost / multiplier, digits + 1)
            else:
                make["%"] = (o.marketPrice - o.averageCost) / o.averageCost * 100
                make["averageCost"] = round(o.averageCost, digits + 1)

            # if short, our profit percentage is reversed
            if o.position < 0:
                make["%"] *= -1
                make["averageCost"] *= -1
                make["marketPrice"] *= -1

            populate.append(make)

        # positions() just returns symbol names, share count, and cost basis.
        # portfolio() returns PnL details and current market prices/values
        df = pd.DataFrame(
            data=populate,
            columns=[
                "type",
                "sym",
                "conId",
                "PC",
                "date",
                "strike",
                "exch",
                "position",
                "averageCost",
                "marketPrice",
                "closeOrder",
                "closeOrderValue",
                "closeOrderProfit",
                "marketValue",
                "totalCost",
                "unrealizedPNLPer",
                "unrealizedPNL",
                "dailyPNL",
                "%",
            ],
        )

        df.sort_values(by=["date", "sym", "PC", "strike"], ascending=True, inplace=True)

        # re-number DF according to the new sort order
        df.reset_index(drop=True, inplace=True)

        allPositions = self.totalFrame(df.copy())
        if allPositions is None:
            logger.info("No current positions found!")
            return None

        desc = "All Positions"
        if self.symbols:
            desc += f" for {', '.join(self.symbols)}"

        printFrame(allPositions, desc)

        # attempt to find spreads by locating options with the same symbol
        symbolCounts = df.pivot_table(index=["type", "sym", "date"], aggfunc="size")

        spreadSyms = set()
        postype: str
        sym: str
        date: str
        for (postype, sym, date), symCount in symbolCounts.items():  # type: ignore
            if postype in {"OPT", "FOP"} and symCount > 1:
                spreadSyms.add((sym, date))

        # print individual frames for each spread since the summations
        # will look better (also sort the symbols so the set() is always in the same order across clients)
        # TODO: we would need a real database of trade intent to record spreads because we can't tell the difference
        #       between a single vertical spread or two vertical spreads combined for math purposes.
        for sym, date in sorted(spreadSyms):
            spread = df[
                df.type.isin({"OPT", "FOP"}) & (df.sym == sym) & (df.date == date)
            ]
            spread = self.totalFrame(spread.copy(), costPrice=True)
            printFrame(spread, f"[{sym}] Potential Spread Identified")

            matchingContracts = [
                contract
                for secType, bqrsym, contract in backQuickRef
                if secType in {"OPT", "FOP"} and bqrsym == sym
            ]

            # transmit the size of the spread only if all are the same
            # TODO: figure out how to do this for butterflies, etc
            equality = 0
            if spread.loc["Total", "position"] == "0":  # yeah, it's a string here
                equality = spread.iloc[0]["position"]

            closeit = self.state.orderPriceForSpread(matchingContracts, equality)
            logger.info("Potential Closing Side: {}", closeit)


@dataclass
class IOpOrders(IOp):
    """Show all currently active orders."""

    symbols: set[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=set)]

    async def run(self):
        # TODO: make new view for these:
        # ords = await self.ib.reqCompletedOrdersAsync(apiOnly=True)

        # We can technically see orders from ALL clients, but it also RE-BINDS orders by canceling
        # them then re-placing them if we run this... so it's not great:
        # also using reqAllOpenOrdersAsync() only gives you a snapshot of the orders at request time,
        # while 'openTrades()' is always kept updated from the gateway notifications.

        # ords = await self.ib.reqAllOpenOrdersAsync()
        # logger.info("Got orders:\n{}", pp.pformat(ords))

        ords = self.ib.openTrades()

        # Note: we extract custom fields here because the default is
        #       returning Order, Contract, and OrderStatus objects
        #       which don't print nicely on their own.
        populate = []

        # save the logs from each order externally so we can print them outside of the summary table
        logs = []

        showDetails = "show" in self.symbols

        # Fields: https://interactivebrokers.github.io/tws-api/classIBApi_1_1Order.html
        # Note: no sort here because we sort the dataframe before showing
        for o in ords:
            # don't print canceled/rejected/inactive orders
            # (unless we have "showDetails" requested, then we print everything
            if o.log[-1].status == "Inactive" and not showDetails:
                continue

            if showDetails:
                logger.info("{}", pp.pformat(o))

            make: dict[str, Any] = {}
            log: dict[str, Any] = {}

            # if symbol filtering requested, only show requested symbols
            if self.symbols and o.contract.symbol not in self.symbols:
                continue

            def populateSymbolDetails(target):
                target["id"] = o.order.orderId
                target["sym"] = o.contract.symbol
                parseContractOptionFields(o.contract, target)
                target["occ"] = (
                    o.contract.localSymbol.replace(" ", "")
                    if isinstance(o.contract, (Option, FuturesOption))
                    else ""
                )

            populateSymbolDetails(make)
            populateSymbolDetails(log)

            make["xchange"] = o.contract.exchange
            make["action"] = o.order.action
            make["orderType"] = o.order.orderType

            # if order type has a sub-strategy, format the name and its arguments
            make["strategy"] = o.order.algoStrategy
            make["params"] = ", ".join(
                [f"{tag.tag}={tag.value}" for tag in o.order.algoParams]
            )

            # IBKR back-populates this on order updates sometimes
            # (IBKR order volatility is expressed in full percent (100% = 100)
            make["vol"] = o.order.volatility if isset(o.order.volatility) else None

            make["qty"] = o.order.totalQuantity
            make["cashQty"] = (
                f"{o.order.cashQty:,.2f}" if isset(o.order.cashQty) else None
            )
            make["lmt"] = f"{o.order.lmtPrice:,.2f}"
            make["aux"] = f"{o.order.auxPrice:,.2f}"
            make["trail"] = (
                f"{o.order.trailStopPrice:,.2f}"
                if isset(o.order.trailStopPrice)
                else None
            )
            make["tif"] = o.order.tif
            # make["oca"] = o.order.ocaGroup
            # make["gat"] = o.order.goodAfterTime
            # make["gtd"] = o.order.goodTillDate
            make["parentId"] = int(o.orderStatus.parentId)
            make["clientId"] = int(o.orderStatus.clientId)
            make["rem"] = o.orderStatus.remaining
            make["filled"] = o.order.totalQuantity - o.orderStatus.remaining
            make["4-8"] = o.order.outsideRth

            # with a bag, we need to calculate a custom pq because each leg can contribute a different amount
            totalMultiplier = 0.0
            if isinstance(o.contract, Bag):
                # is spread, so we need to print more details than just one strike...
                myLegs: list[Any] = []

                for x in o.contract.comboLegs:
                    xcontract = self.state.conIdCache.get(x.conId)

                    # if ID -> Name not in the cache, create it
                    if not xcontract:
                        (xcontract,) = await self.state.qualify(Contract(conId=x.conId))

                    totalMultiplier += float(xcontract.multiplier or 1)

                    # now the name will be in the cache!
                    lsym = self.state.conIdCache[x.conId].localSymbol
                    lsymsym, *lsymrest = lsym.split()

                    # TODO: fix date format for futures
                    #   File "/Users/matt/repos/tplat/icli/icli/lang.py", line 3815, in run
                    #    lsymrest[-9],  # 2
                    #    └ 'P5600'

                    # IndexError: string index out of range
                    if isinstance(xcontract, FuturesOption):
                        lsymrest = None

                    # leg of spread is an IBKR OPTION SYMBOL (SPY 20240216C00500000)
                    if lsymrest:
                        lsymrest = lsymrest[0]
                        # fmt: off
                        myLegs.append(
                            (
                                x.action[0],  # 0
                                x.ratio,  # 1
                                # Don't need symbol because row has symbol...
                                # lsymsym,
                                lsymrest[-9],  # 2
                                str(dateutil.parser.parse("20" + lsymrest[:6]).date()),
                                round(int(lsymrest[-8:]) / 1000, 2),  # 4
                                lsym.replace(" ", ""),  # 5
                            )
                        )
                        # fmt: on
                    else:
                        # else, leg of spread is a single STOCK SYMBOL (SPY)
                        myLegs.append(
                            (
                                x.action[0],  # 0
                                x.ratio,  # 1
                                "",  # 2
                                "",  # 3
                                0,  # 4
                                lsym.replace(" ", ""),  # 5
                            )
                        )
                else:
                    # normalize all multipliers in the bags by their total weight for the final credit/debit value calculation
                    totalMultiplier /= len(o.contract.comboLegs)

                # if all P or C, make it top level
                if all(l[2] == myLegs[0][2] for l in myLegs):
                    make["PC"] = myLegs[0][2]

                # if all same date, make it top level
                if all(l[3] == myLegs[0][3] for l in myLegs):
                    make["date"] = myLegs[0][3]

                # if all same strike, make it top level
                if all(l[4] == myLegs[0][4] for l in myLegs):
                    make["strike"] = myLegs[0][4]

                make["legs"] = self.state.nameForContract(o.contract)

            # extract common fields for re-use below
            multiplier = totalMultiplier or float(o.contract.multiplier or 1)
            lmtPrice = float(o.order.lmtPrice or 0)
            auxPrice = float(o.order.auxPrice or 0)
            totalQuantity = float(o.order.totalQuantity)

            # auxPrice overrides lmtPrice if both exists (e.g. for stops, IBKR seems to auto-fill the limit price
            # as the near-term market price even though it isn't used. Only the aux price is used for stops, and other
            # order types have lmtPrice but no auxPrice, so this should work...)
            pq = (auxPrice or lmtPrice) * totalQuantity * multiplier
            make["lreturn"] = 0
            make["lcost"] = 0

            # record whether this order value is a credit into or debit from the account
            if o.order.action == "SELL":
                # IBKR 'sell' prices are always positive and represents a credit back to the account when executed.
                # We must have at least one populated lmt or aux price for a sell to be valid.
                # (though, MKT orders have a zero price, so those are still okay)
                assert (lmtPrice > 0 or auxPrice > 0) or (
                    lmtPrice == 0 and o.order.orderType == "MKT"
                ), f"How is your order trigger price negative? Order: {o.order}"

                make["lreturn"] = pq
            else:
                # else, if not SELL, then it must be BUY
                assert (
                    o.order.action == "BUY"
                ), f"How did you get a non-BUY AND non-SELL order action? Got: {o.order}"

                # buying is a cost (debit) unless the buy has a negative limit price, then it's a credit.
                # if buying for a negative price, record value of "negative purchase" as "lreturn" and not "lcost"
                if lmtPrice < 0:
                    # QUESTION/TEST: if we are going short, is the quantity zero AND the lmtPrice is negative, so this cancels out to negative again?
                    make["lreturn"] = -pq
                else:
                    # else, limit price is > 0 so it's a debit/cost/charge to us
                    make["lcost"] = pq

            log["logs"] = o.log

            populate.append(make)
            logs.append(log)

        # fmt: off
        df = pd.DataFrame(
                data=populate,
                columns=["id", "parentId", "clientId", "action", "sym", 'PC', 'date', 'strike',
                    "xchange", "orderType", "strategy", "params", "vol",
                    "qty", "cashQty", "filled", "rem", "lmt", "aux", "trail", "tif",
                    "4-8", "lreturn", "lcost", "occ", "legs"],
                )
        # fmt: on

        if df.empty:
            logger.info(
                "{}No open orders exist for client id {}!",
                f"[{', '.join(sorted(self.symbols))}] " if self.symbols else "",
                self.state.clientId,
            )
            return

        df.sort_values(
            by=["date", "sym", "strike", "qty", "action"],
            ascending=True,
            inplace=True,
        )

        df = df.set_index("id")
        fmtcols = ["lreturn", "lcost"]
        df[fmtcols] = df[fmtcols].astype(str)

        # logger.info("Types are: {}", df.info())
        df = addRowSafe(df, "Total", df[fmtcols].astype(float).sum(axis=0))

        df = df.fillna("")

        df.loc[:, fmtcols] = df[fmtcols].astype(float).map(lambda x: f"{x:,.2f}")

        toint = ["qty", "filled", "rem", "clientId", "parentId"]
        df[toint] = df[toint].map(lambda x: f"{x:,.0f}" if x else "")
        df[["4-8"]] = df[["4-8"]].map(lambda x: True if x else "")

        # print the status logs for each current order...

        for log in logs:
            logger.info("[{} :: {} :: {}] EVENT LOG", log["id"], log["sym"], log["occ"])

            for l in log["logs"]:
                logger.info(
                    "[{} :: {} :: {}] {}: {} — {}",
                    log["id"],
                    log["sym"],
                    log["occ"],
                    l.time,
                    l.status,
                    l.message,
                )

        # now print the actual open orders
        printFrame(df)


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
        printFrame(df.convert_dtypes(), f"Groups Summary")


@dataclass
class IOpExecutions(IOp):
    """Display all executions including commissions and PnL."""

    symbols: set[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*symbols",
                desc="Optional symbols to filter for in result listings",
                convert=lambda x: set([s.upper() for s in x]),
            )
        ]

    async def run(self):
        # "Fills" has:
        # - contract
        # - execution (price at exchange)
        # - commissionReport (commission and PNL)
        # - time (UTC)
        # .executions() is the same as just the 'execution' value in .fills()

        # the live updating only works for the current client activity, so to fetch _all_
        # executions for the entire user over all clients, run "exec refresh" or "exec update"
        # ALSO NOTE: we stopped fetching executions on startup too, so this will restore previous
        #            executions if you restart your client in the middle of a trading session.
        REFRESH_CHECK = {":REFRESH", ":UPDATE", ":U", ":R", ":UPD", ":REF"}
        SELF_CHECK = {":SELF", ":MINE", ":S", ":M"}

        assert (
            REFRESH_CHECK & SELF_CHECK == set()
        ), "There are conflicting keywords in REFRESH_CHECK vs SELF_CHECK?"

        if REFRESH_CHECK & self.symbols:
            self.symbols -= REFRESH_CHECK
            await self.state.loadExecutions()

        fills = self.ib.fills()
        # logger.info("Fills: {}", pp.pformat(fills))
        contracts = []
        executions = []
        commissions = []
        for f in fills:
            contracts.append(f.contract)
            executions.append(f.execution)
            commissions.append(f.commissionReport)

        if False:
            logger.info(
                "C: {}\nE: {}: CM: {}",
                pp.pformat(contracts),
                pp.pformat(executions),
                pp.pformat(commissions),
            )

        use = []
        for name, l in [
            ("Contracts", contracts),
            ("Executions", executions),
            ("Commissions", commissions),
        ]:
            df = pd.DataFrame(l)  # type: ignore
            if df.empty:
                logger.info("No {}", name)
            else:
                use.append((name, df))

        if not use:
            return None

        df = pd.concat({name: frame for name, frame in use}, axis=1)

        # Goodbye multiindex...
        df.columns = df.columns.droplevel(0)

        # enforce the dataframe is ordered from oldest executions to newest executions as defined by full original timestamp order.
        df.sort_values(by=["time", "clientId"], inplace=True)

        # show only executions for the current client id
        # TODO: allow showing only from a specific client id provided as a paramter too?
        if SELF_CHECK & self.symbols:
            self.symbols -= SELF_CHECK
            df = df[df.clientId == self.state.clientId]

        # if symbol filter requested, remove non-matching contracts.
        # NOTE: we filter on SYMBOL and not "localSymbol" so we don't currently match on extended details like OCC symbols.
        if self.symbols:
            df = df[df.symbol.isin(self.symbols)]

        # clean up any non-matching values due to symbols filtering
        if self.symbols:
            df = df[df.conId.notna()]

        if df.empty:
            logger.info(
                "[{}] No execution history found!", " ".join(sorted(self.symbols))
            )
            return False

        # Remove duplicate columns...
        df = df.loc[:, ~df.columns.duplicated()]

        # convert to Eastern time and drop date (since these reports only show up for one day, it's all duplicate details)
        df["time"] = df["time"].apply(pd.Timestamp).dt.tz_convert("US/Eastern")
        df["timestamp"] = df["time"]

        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        df["time"] = df["time"].dt.strftime("%H:%M:%S")

        df["c_each"] = df.commission / df.shares

        tointActual = ["clientId", "orderId"]
        df[tointActual] = df[tointActual].map(lambda x: f"{int(x)}" if x else "")

        # really have to stuff this multiplier change in there due to pandas typing requirements
        df.multiplier = df.multiplier.replace("", 1).fillna(1).astype(float)

        df["total"] = round(df.shares * df.avgPrice, 2) * df.multiplier

        # Note: 'realizedPNL' for the closing transactions *already* includes commissions for both the buy and sell executions,
        #       so *don't* subtract commissions again anywhere.
        df["dayProfit"] = df.realizedPNL.cumsum()

        df["RPNL_each"] = df.realizedPNL / df.shares

        # provide a weak calculation of commission as percentage of PNL and of traded value.
        # Note: we estimate the entry commission by just doubling the exit commission (which isn't 100% accurate, but
        #       if the opening trade was more than 1 day ago, we don't have the opening matching executions to read
        #       the execution from (and we aren't keeping a local fullly logged execution history, but maybe we should
        #       add logged execution history as a feature in the future?)
        df["c_pct"] = df.commission / (df.total + df.commission)
        df["c_pct_RPNL"] = (df.commission * 2) / (df.realizedPNL + (df.commission * 2))

        dfByTrade = df.groupby("orderId side localSymbol".split()).agg(  # type: ignore
            dict(
                date=["min"],
                time=[("start", "min"), ("finish", "max")],  # type: ignore
                price=["mean"],
                shares=["sum"],
                total=["sum"],
                commission=["sum"],
                # TODO: we need to calculate a manlal c_each by (total commision / shares) instead of mean of the c_each directly
                c_each=["mean"],
            )
        )

        # also show if the order occurred via multiple executions over time
        # (single executions will have zero duration, etc)
        dfByTrade["duration"] = pd.to_datetime(
            dfByTrade["time"]["finish"], format="%H:%M:%S"
        ) - pd.to_datetime(dfByTrade["time"]["start"], format="%H:%M:%S")

        # convert the default pandas datetime difference just into a number of seconds per row
        # (the returned "Series" from the subtraction above doesn't allow .seconds to be applied
        #  as a column operation, so we apply it row-element-wise here instead)
        dfByTrade["duration"] = dfByTrade.duration.apply(lambda x: x.seconds)

        # also show commission percent for traded value per row
        dfByTrade["c_pct"] = dfByTrade.commission / (
            dfByTrade.total + dfByTrade.commission
        )

        dfByTimeProfit = df.copy().sort_values(
            by=["time", "orderId", "secType", "side", "localSymbol"]
        )

        needsPrices = "price shares total commission c_each".split()
        dfByTrade[needsPrices] = dfByTrade[needsPrices].map(fmtPrice)

        # this currently has a false pandas warning about "concatenation with empty or all-NA entries is deprecated"
        # but nothing is empty or NA in these columns. Their logic for checking their warning condition is just broken.
        # (or their "FutureWarning" error message is so bad we can't actually see what the problem is)
        df = addRowSafe(df, "sum", df[["shares", "price", "commission", "total"]].sum())

        df = addRowSafe(
            df,
            "sum-buy",
            df[["shares", "price", "commission", "total"]][df.side == "BOT"].sum(),
        )

        df = addRowSafe(
            df,
            "sum-sell",
            df[["shares", "price", "commission", "total"]][df.side == "SLD"].sum(),
        )

        df.loc["profit", "total"] = (
            df.loc["sum-sell"]["total"] - df.loc["sum-buy"]["total"]
        )
        df.loc["profit", "price"] = (
            df.loc["sum-sell"]["price"] - df.loc["sum-buy"]["price"]
        )

        eachSharePrice = ["c_each", "shares", "price"]
        df = addRowSafe(df, "med", df[eachSharePrice].median())
        df = addRowSafe(df, "mean", df[eachSharePrice].mean())

        needsPrices = "c_each shares price avgPrice commission realizedPNL RPNL_each total dayProfit c_pct c_pct_RPNL".split()
        df[needsPrices] = df[needsPrices].map(fmtPrice)

        # convert contract IDs to integers (and fill in any missing
        # contract ids with placeholders so they don't get turned to
        # strings with the global .fillna("") below).
        df.conId = df.conId.fillna(-1).astype(int)

        # new pandas strict typing doesn't allow numeric columns to become "" strings, so now just
        # convert ALL columns to a generic object type
        df = df.astype(object)

        # display anything NaN as empty strings so it doesn't clutter the interface
        df.fillna("", inplace=True)

        df.rename(columns={"lastTradeDateOrContractMonth": "conDate"}, inplace=True)
        # ignoring: "execId" (long string for execution recall) and "permId" (???)

        # removed: lastLiquidity avgPrice
        df = df[
            (
                """clientId secType conId symbol conDate right strike date exchange tradingClass localSymbol time orderId
         side  shares  cumQty price    total realizedPNL RPNL_each
         commission c_each c_pct c_pct_RPNL dayProfit""".split()
            )
        ]

        dfByTimeProfit.set_index("timestamp", inplace=True)

        dfByTimeProfit["profit"] = dfByTimeProfit.where(dfByTimeProfit.realizedPNL > 0)[
            "realizedPNL"
        ]

        dfByTimeProfit["loss"] = dfByTimeProfit.where(dfByTimeProfit.realizedPNL < 0)[
            "realizedPNL"
        ]

        # using "PNL == 0" to determine an open order should work _most_ of the time, but if you
        # somehow get exactly a $0.00 PNL on a close, it will be counted incorrectly here.
        dfByTimeProfit["opening"] = dfByTimeProfit.where(
            dfByTimeProfit.realizedPNL == 0
        )["orderId"]
        dfByTimeProfit["closing"] = dfByTimeProfit.where(
            dfByTimeProfit.realizedPNL != 0
        )["orderId"]

        # ==============================================
        # Profit by Half Hour (split out By Day) Control
        # ==============================================

        # Function to assign each timestamp to its corresponding trading day
        # Trading day is defined as 6PM ET to 5PM ET the next day
        def assign_trading_day(timestamp):
            # Convert timestamp to Eastern Time if it's not already
            if (
                timestamp.tzinfo is None
                or timestamp.tzinfo.utcoffset(timestamp) is None
            ):
                # Assumes the timestamp is already in ET, adjust if needed
                et_timestamp = timestamp
            else:
                # If timestamp has timezone info, convert to ET
                eastern = pytz.timezone("US/Eastern")
                et_timestamp = timestamp.astimezone(eastern)

            # If time is before 6PM, it belongs to the previous day's trading session
            if et_timestamp.hour < 18:  # Before 6PM ET
                trading_day = et_timestamp.date()
            else:  # 6PM ET or later
                trading_day = et_timestamp.date() + pd.Timedelta(days=1)

            return pd.Timestamp(trading_day)

        # Apply the trading day logic to your dataframe
        dfByTimeProfit["trading_day"] = dfByTimeProfit.index.map(assign_trading_day)

        # Group by trading day and then resample by 30 minutes
        trading_day_groups = dfByTimeProfit.groupby("trading_day")

        # Create an empty dataframe to store results
        all_results = pd.DataFrame()

        # Process each trading day separately, printing one table per trading day
        desc = ""
        if self.symbols:
            desc = f" Filtered for: {', '.join(self.symbols)}"

        # Print the original execution summary
        printFrame(df.convert_dtypes(), f"Execution Summary{desc}")

        # Create the daily summary table to show at the end
        trading_day_summary = dfByTimeProfit.groupby("trading_day").agg(
            dict(
                realizedPNL="sum",
                orderId=[("orders", "nunique"), ("executions", "count")],
                profit="count",
                loss="count",
                opening="nunique",
                closing="nunique",
            )
        )

        # Process each trading day and print tables individually
        for trading_day, group in trading_day_groups:
            # Resample the group by 30 minutes
            day_profit = group.resample("30Min").agg(
                dict(
                    realizedPNL="sum",
                    orderId=[("orders", "nunique"), ("executions", "count")],
                    profit="count",
                    loss="count",
                    opening="nunique",
                    closing="nunique",
                )
            )

            # Calculate cumulative sum for just this trading day
            day_profit["dayProfit"] = day_profit.realizedPNL.cumsum()

            # Format prices
            needsPrices = "realizedPNL dayProfit".split()
            day_profit[needsPrices] = day_profit[needsPrices].map(fmtPrice)

            # Add sum row for this day's table
            numeric_cols = "orderId profit loss opening closing".split()
            day_profit.loc["sum"] = day_profit.loc[:, numeric_cols].sum()

            # Format the trading day for display
            trading_day_str = pd.Timestamp(trading_day).strftime("%Y-%m-%d")

            # Print separate table for each trading day
            printFrame(
                day_profit.convert_dtypes(),
                f"Profit by Half Hour for Trading Day {trading_day_str}{desc}",
            )

        # Format the daily summary
        trading_day_summary["totalProfit"] = trading_day_summary.realizedPNL
        trading_day_summary[["realizedPNL", "totalProfit"]] = trading_day_summary[
            ["realizedPNL", "totalProfit"]
        ].map(fmtPrice)

        # Print the trading day summary at the end
        printFrame(trading_day_summary.convert_dtypes(), f"Profit by Trading Day{desc}")

        # ==============================================
        # END Profit by Half Hour (split out By Day) Control
        # ==============================================

        if False:
            # (original "profitByHour" (actually by half hour) implementation that didn't break out multiple days
            profitByHour = dfByTimeProfit.resample("30Min").agg(  # type: ignore
                dict(
                    realizedPNL="sum",
                    orderId=[("orders", "nunique"), ("executions", "count")],  # type: ignore
                    profit="count",
                    loss="count",
                    opening="nunique",
                    closing="nunique",
                )
            )

            profitByHour["dayProfit"] = profitByHour.realizedPNL.cumsum()

            needsPrices = "realizedPNL dayProfit".split()
            profitByHour[needsPrices] = profitByHour[needsPrices].map(fmtPrice)

            desc = ""
            if self.symbols:
                desc = f" Filtered for: {', '.join(self.symbols)}"

            printFrame(df.convert_dtypes(), f"Execution Summary{desc}")

            profitByHour.loc["sum"] = profitByHour.loc[
                :, "orderId profit loss opening closing".split()
            ].sum()

            printFrame(profitByHour.convert_dtypes(), f"Profit by Half Hour{desc}")

        printFrame(
            dfByTrade.sort_values(
                by=[("date", "min"), ("time", "start"), "orderId", "localSymbol"]  # type: ignore
            ).convert_dtypes(),
            f"Execution Summary by Complete Order{desc}",
        )

        # count orderId values in the multi-index:
        # (IBKR complains if you have more than ((orders executed + 1) * 20) order submissions per day)
        # logger.info("Orders executed: {}", len(dfByTrade.groupby(level=[0, 0]).size()))


@dataclass
class IOpQualify(IOp):
    """Qualify contracts in the cache for future usage.

    Current reason this exists: IBKR apparently refuses to qualify 0dte /ES contracts, so we
    look them *all* up the day before so we can access them live the next day."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: sorted(expand_symbols(x)))]

    async def run(self):
        # build contract objects from names
        logger.info("Qualifying for names: {}", self.symbols)

        contracts = [contractForName(s) for s in self.symbols]
        logger.info("Qualifying: {}", contracts)

        # ask our cache or IBKR for the complete details including unique contract ids
        got = await self.state.qualify(*contracts)

        # results
        logger.info("Result: {}", got)
        return got


@dataclass
class IOpQuotesAdd(IOp):
    """Add live quotes to display."""

    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: expand_symbols(x))]

    async def run(self):
        keys = await self.state.addQuotes(self.symbols)

        if keys:
            # also update current client persistent quote snapshot
            # (only if we added new quotes...)
            await self.runoplive("qsnapshot")

        return keys


@dataclass
class IOpQuotesAlign(IOp):
    """Add a group of commonly used together quotes and spreads all at once.

    This commands adds:
      - ATM strangle
      - ATM ±pts straddle
      - call spread +pts width
      -  put spread -pts width
    """

    symbol: str = field(init=False)
    points: float = field(init=False)
    width: float = field(init=False)
    tradingClass: str = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg("points", convert=float, default=10),
            DArg("width", convert=float, default=20),
            DArg(
                "tradingClass",
                default="",
                desc="If you need to disambiguate your contracts, you can add a custom trading class. Default: unused",
            ),
        ]

    @property
    def strikeWidthOffset(self) -> float:
        """Offset against the starting points for width calculations.

        e.g. a straddle is points=0, width=0
             a strangle is points=N, width=K, but our commands take (points from ATM) (width from ATM),
               so we need to provide (points + width) for as:
               points=10 width=20 gives the command ATM+10, ATM+30 for the strikes."""
        return self.width + self.points

    async def run(self):
        logger.info(
            "[{}] Using ATM width: {} and strike width: {}",
            self.symbol,
            self.points,
            self.strikeWidthOffset - self.points,
        )

        tradingClassExtension = f"-{self.tradingClass}" if self.tradingClass else ""

        # strangle
        # e.g: straddle /ES 0
        a = self.runoplive("straddle", f"{self.symbol}{tradingClassExtension} 0")

        # straddle
        # e.g.: straddle /ES 10
        b = self.runoplive(
            "straddle", f"{self.symbol}{tradingClassExtension} {self.points}"
        )

        # put spread
        # e.g.: for points=10, width=20 == straddle /ES v p -10 -30
        c = self.runoplive(
            "straddle",
            f"{self.symbol}{tradingClassExtension} v p -{self.points} -{self.strikeWidthOffset}",
        )

        # call spread
        # e.g.: for points=10, width=20 == straddle /ES v c 10 30
        d = self.runoplive(
            "straddle",
            f"{self.symbol}{tradingClassExtension} v c {self.points} {self.strikeWidthOffset}",
        )

        await asyncio.gather(*[a, b, c, d])

        # also update current client persistent quote snapshot
        await self.runoplive("qsnapshot")


@dataclass
class IOpQuotesAddFromOrderId(IOp):
    """Add symbols for current orders to the quotes view."""

    orderIds: list[int] = field(init=False)

    def argmap(self):
        return [DArg("*orderIds", lambda xs: [int(x) for x in xs])]

    async def run(self):
        trades = self.ib.openTrades()

        if not self.orderIds:
            addTrades = trades
        else:
            addTrades = []
            for oid in self.orderIds:
                useTrade = None
                for t in trades:
                    if t.order.orderId == oid:
                        useTrade = t
                        break

                if not useTrade:
                    logger.error("No order found for id {}", oid)
                    continue

                addTrades.append(useTrade)

        # if we have no orders to add, don't do anything
        if not addTrades:
            return

        for useTrade in addTrades:
            # If this is a new session and we haven't previously cached the
            # contract id -> name mappings, we need to look them all up now
            # or else the next print of the quote toolbar will throw lots
            # of missing key exceptions when trying to find names.
            if useTrade.contract.comboLegs:
                # TODO: verify this logic still holds after the contract cache refactoring
                for x in useTrade.contract.comboLegs:
                    # if ID -> Name not in the cache, create it
                    if x.conId not in self.state.conIdCache:
                        await self.state.qualify(Contract(conId=x.conId))
            else:
                if useTrade.contract.conId not in self.state.conIdCache:
                    await self.state.qualify(Contract(conId=useTrade.contract.conId))

            self.state.addQuoteFromContract(useTrade.contract)

        # also update current client persistent quote snapshot
        await self.runoplive("qsnapshot")


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
                            "{}{}{}{:0>8}".format(
                                self.rename or self.symbol,
                                date[2:],
                                pc,
                                int(strike * 1000),
                            )
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


@dataclass
class IOpPreQualify(IOp):
    """IBKR refuses to qualify valid FOP contracts on the day they expire, but we can cache them ahead of time."""

    symbol: str = field(init=False)
    days: int = field(init=False)
    overwrite: bool = field(init=False)
    verbose: bool = field(init=False)
    tradingClass: str = field(init=False)

    def argmap(self):
        return [
            DArg("symbol", convert=str.upper),
            DArg(
                "days",
                convert=int,
                default=15,
            ),
            DArg("verbose", default=False, convert=lambda x: bool(x)),
            DArg("overwrite", default=False, convert=lambda x: bool(x)),
            DArg(
                "tradingClass",
                default="",
                desc="If you need to disambiguate your contracts, you can add a custom trading class. Default: unused",
            ),
        ]

    async def run(self):
        # Steps:
        #  1. Collect option chain for symbol
        #  2. For each date in chain, and for each strike in each chain, qualify the contract into our cache.

        now = whenever.ZonedDateTime.now("US/Eastern")
        nowdate = now.date()

        highestdate = now.add(
            whenever.days(self.days), disambiguate="compatible"
        ).date()

        symbol = self.symbol
        chains = await self.runoplive("chain", symbol)

        for sym, chain in chains.items():
            logger.info(
                "[{}] Qualifying contracts for dates between {} and {}: {}",
                sym,
                nowdate,
                highestdate,
                chain.keys(),
            )

            for date, strikes in chain.items():
                pdate = whenever.LocalDateTime.from_py_datetime(
                    dateutil.parser.parse(date)
                ).date()

                # don't qualify the past...
                if pdate < nowdate:
                    continue

                # don't qualify expired contracts if our cached chains have old dates
                if pdate > highestdate:
                    continue

                # date format is YYMMDD so remove the first '20'
                datefmt = date[2:]
                names = []
                tradingClassExtension = (
                    f"-{self.tradingClass}" if self.tradingClass else ""
                )
                for strike in strikes:
                    # Note: we use the INPUT symbol because it will have the proper type designator.
                    # (e.g. we need /ES... and not ES... and 'sym' from the cache is the underlying symbol. We should probably fix that too to include the contract type)
                    names.extend(
                        [
                            f"{symbol}{datefmt}P{int(strike * 1000):08}{tradingClassExtension}",
                            f"{symbol}{datefmt}C{int(strike * 1000):08}{tradingClassExtension}",
                        ]
                    )

                # run qualification batch at each date instead of just runnin them ALL at the end
                logger.info(
                    "[{} :: {}] Qualifying {} contracts (overwrite: {})...",
                    symbol,
                    date,
                    len(names),
                    self.overwrite,
                )

                got = await self.state.qualify(
                    *[contractForName(name) for name in names], overwrite=self.overwrite
                )

                if self.verbose:
                    logger.info("{}", got)


@dataclass
class IOpWhen(IOp):
    """Run a command when a symbol price condition matches.

    WHEN AAPL > 200 BUY AAPL $10_000 AF
    """

    symbol: str = field(init=False)
    action: str = field(init=False)
    price: str = field(init=False)
    cmd: list[str] = field(init=False)

    def argmap(self):
        return [DArg("symbol"), DArg("action"), DArg("price"), DArg("*cmd")]

    async def run(self):
        """Construct and save live datastructure for per-quote checking.

        Example:

        actions[symbol] = [(price, action, cmd)]

        Then on each quote update, we read actions[symbol], find any matching price details (we may
        need additional state if we are tracking crossovers because we need before/after checks) then
        when triggered, run the command (or preview the command, but 'preview' can be part of command itself too).
        """

        raise NotImplementedError


@dataclass
class IOpQuoteSave(IOp):
    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        self.cache.set(cacheKey, set(self.symbols))  # type: ignore
        logger.info("[{}] {}", self.group, self.symbols)

        repopulate = [f'"{x}"' for x in self.symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )


@dataclass
class IOpQuoteSaveClientSnapshot(IOp):
    """Save the current live quote state for THIS CLIENT ID only so it auto-reloads on the next startup."""

    def argmap(self):
        return []

    async def run(self):
        cacheKey = ("quotes", f"client-{self.state.clientId}")
        allLiveContracts = [c.contract for c in self.state.quoteState.values()]
        self.cache.set(cacheKey, {"contracts": allLiveContracts})  # type: ignore

        # This log line is nice for debugging but too noisy to run on every snapshot
        # since every 'add' or 'oadd' is a new snapshot saving event too.
        if False:
            logger.info(
                "[{}] Saved {} contract ids for snapshot: {}",
                cacheKey,
                len(allLiveContracts),
                sorted(
                    [
                        (c.contract.localSymbol, c.contract.conId)
                        for c in self.state.quoteState.values()
                    ]
                ),
            )


@dataclass
class IOpColors(IOp):
    """Select a new color scheme either by collection name or with direct colors."""

    style: str = field(init=False)

    def argmap(self):
        return [DArg("style")]

    async def run(self):
        cacheKey = ("colors", f"client-{self.state.clientId}")
        cacheVal = dict(toolbar=self.style)
        self.cache.set(cacheKey, {"colors": cacheVal})  # type: ignore

        self.state.updateToolbarStyle(self.style)


@dataclass
class IOpColorsLoad(IOp):
    otherClientId: int = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*otherClientId",
                convert=lambda x: int(x) if x else self.state.clientId,
                desc="Optional Client ID if loading another client's color setting",
            )
        ]

    async def run(self):
        cacheKey = ("colors", f"client-{self.otherClientId}")
        cons = self.cache.get(cacheKey)  # type: ignore
        if not cons:
            return False

        try:
            # snapshots are always saved with exact Contract objects, so we can just restore them directly
            cs = cons.get("colors", {}).get("toolbar")
            self.state.updateToolbarStyle(cs)
            return True
        except:
            pass

        return False


@dataclass
class IOpQuoteAppend(IOp):
    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            logger.error(
                "[{}] No quote group found. Creating new quote group!", self.group
            )
            symbols = set()

        self.cache.set(cacheKey, symbols | set(self.symbols))  # type: ignore
        repopulate = [f'"{x}"' for x in self.symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )


@dataclass
class IOpQuoteRemove(IOp):
    """Remove symbols from a quote group."""

    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        nocache = False
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            nocache = True
            symbols = self.state.quoteState
            logger.error(
                "[{}] No quote group found so using live quote list...", self.group
            )

        goodbye = set()
        for s in self.symbols:
            for symbol, ticker in symbols.items():
                # guard against running fnmatch on the tuple entries we use for spread quotes
                # (spread quotes can only be removed by :N id)
                if isinstance(symbol, str):
                    if fnmatch.fnmatch(symbol, s):
                        logger.info("Dropping quote: {}", symbol)
                        goodbye.add((symbol, ticker.contract.conId))

        # don't *CREATE* the cache key if we didn't use the cache anyway
        if not nocache:
            symbols -= {x[0] for x in goodbye}  # type: ignore
            self.cache.set(cacheKey, {x[1] for x in goodbye})  # type: ignore

        if not goodbye:
            logger.warning("No matching symbols found?")
            return

        goodbyeIds = [f"{conId}" for _symbol, conId in goodbye]
        logger.info(
            "Removing quotes: {}",
            ", ".join([f"{symbol} -> {conId}" for symbol, conId in goodbye]),
        )

        logger.info("rm {}", " ".join(goodbyeIds))

        await self.runoplive("remove", " ".join(goodbyeIds))


@dataclass
class IOpQuoteRestore(IOp):
    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group")]

    async def run(self) -> bool:
        """Returns True if we restored quotes, False if no quotes were restored.

        The return value is used to determine whether we load the "default" quote set on startup too.
        """
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return False

        repopulate = [f'"{x}"' for x in symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )

        return True


@dataclass
class IOpQuoteLoadSnapshot(IOp):
    otherClientId: int = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*otherClientId",
                convert=lambda x: int(x) if x else self.state.clientId,
                desc="Optional Client ID if loading another client's snapshot",
            )
        ]

    async def run(self):
        cacheKey = ("quotes", f"client-{self.otherClientId}")
        cons = self.cache.get(cacheKey)  # type: ignore
        if not cons:
            # if no ids, just don't do anything.
            # also don't bother with any status/warning message because this runs on startup
            # and we don't need to know if we didn't restore anything.
            return False

        try:
            # snapshots are always saved with exact Contract objects, so we can just restore them directly
            cs = cons.get("contracts", [])
            for c in cs:
                self.state.addQuoteFromContract(c)

            logger.info("Restored {} quotes from snapshot", len(cs))

            return True
        except:
            # logger.exception("Failed?")
            # format is incorrect, just ignore it
            pass

        return False


@dataclass
class IOpQuoteClean(IOp):
    group: str = field(init=False)

    def argmap(self):
        return [DArg("group")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return

        # Find any expired option symbols and remove them
        remove = []
        now = whenever.ZonedDateTime.now("US/Eastern")

        # if after market close, use today; else use previous day since market is still open
        if (now.hour, now.minute) < (16, 15):
            now = now.subtract(whenever.days(1), disambiguate="compatible")

        datecompare = f"{now.year - 2000}{now.month:02}{now.day:02}"
        for x in symbols:
            if len(x) > 10:
                date = x[-15 : -15 + 6]
                if date <= datecompare:
                    logger.info("Removing expired quote: {}", x)
                    remove.append(f'"{x}"')

        # TODO: fix bug where it's not translating SPX -> SPXW properly for the live removal
        await self.runoplive(
            "qremove",
            "global " + " ".join(remove),
        )


@dataclass
class IOpQuoteList(IOp):
    """Show current quote group names usable by other q* commands"""

    groups: set[str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*groups",
                convert=set,
                desc="optional groups to fetch. if not provided, return all groups and their members.",
            )
        ]

    async def run(self):
        if self.groups:
            # if only specific groups requested, use them
            groups = sorted(self.groups)
        else:
            # else, use all quote groups found
            groups = sorted([k[1] for k in self.cache if k[0] == "quotes"])  # type: ignore

        logger.info("Groups: {}", pp.pformat(groups))

        for group in groups:
            found = self.cache.get(("quotes", group), [])  # type: ignore

            if isinstance(found, list):
                found = sorted(found)

            # printing all contract IDs directly can make it easy to copy all current quotes
            # to another system or sharing a quote view with people who don't have your saved
            # quote database.
            if isinstance(found, dict) and "contracts" in found:
                contracts = found["contracts"]
                contracts = sorted(contracts, key=lambda x: x.localSymbol)

                logger.info(
                    "[{}] Members (conIds): {}",
                    group,
                    " ".join([str(x.conId) for x in contracts]),
                )

            logger.info("[{}] Members: {}", group, pp.pformat(found))


@dataclass
class IOpQuoteGroupDelete(IOp):
    """Delete an entire quote group"""

    groups: set[str] = field(init=False)

    def argmap(self):
        return [DArg("*groups", convert=set, desc="quote group names to delete.")]

    async def run(self):
        if not self.groups:
            logger.error("No groups provided!")

        for group in sorted(self.groups):
            logger.info("Deleting quote group: {}", group)
            self.cache.delete(("quotes", group))  # type: ignore


# ===========================================================================
# PREDICATE MANAGEMENT SECTION
# ===========================================================================
@dataclass
class IOpPredicateCreate(IOp):
    predicate: str = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*predicate",
                convert=lambda x: " ".join(x).strip(),
                verify=lambda x: bool(x) and len(x) > 10,
            )
        ]

    async def run(self):
        logger.info("[{}] Configuring predicate...", self.predicate)

        # We need to:
        #   - record predicate with target operations into state we can read/delete
        #   - attach live price/algo fetcher to predicate so it can check itself on every ticker update
        #   - subscribe predicate to symbols used for decision making
        #   - view current state of each element of the predicate

        pid = self.state.ifthenRuntime.parse(self.predicate)

        prepredicate = self.state.ifthenRuntime[pid]
        logger.info("[{}] Parsed: {}", pid, pp.pformat(prepredicate))

        assert prepredicate is not None

        await self.state.predicateSetup(prepredicate)

        # now, since we attached the proper data extractors, we can enable the predicate for running
        # Here, for SINGLE predicates, we mark them to delete after one success.
        self.state.ifthenRuntime.activate(pid, once=True)


@dataclass
class IOpPredicateList(IOp):
    """List all actively running predicates with their runtime IDs"""

    pids: list[int] = field(init=False)

    def argmap(self):
        return [DArg("*pids", convert=lambda x: [int(y) for y in x])]

    async def run(self):
        preds, actives = self.state.ifthenRuntime.report()

        if self.pids:
            for p in self.pids:
                logger.info(
                    "[{}] Found ({}): {}",
                    p,
                    "ACTIVE" if p in actives else "NOT ACTIVE",
                    pp.pformat(preds.get(p)),
                )
        else:
            logger.info("Active Predicates ({}): {}", len(actives), sorted(actives))
            logger.info("All Predicates ({}):", len(preds))
            logger.info("{}", pp.pformat(preds))


@dataclass
class IOpPredicateDelete(IOp):
    """Delete one or more predicates by their current session id (visible at creation time or from iflist)."""

    ids: set[int] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*ids",
                convert=lambda x: set(map(int, x)),
                desc="Predicate IDs to delete (from iflist)",
            )
        ]

    async def run(self):
        logger.info("Deleting Predicates: {}", sorted(self.ids))
        for pid in self.ids:
            try:
                predicate = self.state.ifthenRuntime.remove(pid)
                logger.info("[{}] predicate deleted: {}", pid, pp.pformat(predicate))
            except:
                logger.warning("[{}] predicate not found; nothing deleted", pid)


@dataclass
class IOpPredicateGroup(IOp):
    """Manage a predicate group (or tree) structure embedded in one command.

      Purpose: Allow in-cli creation of OCA and OTO and OTOCO if/then trees.

      Basically we need to:
        - create various named ifthen predicates
        - attach named predicates to either OTO (active -> waiting) or OCA (peers) groups
        - we can also attach OTO or OCA groups to other OTO or OCA groups as well for continually recursive soultions (e.g. short -> long -> short -> long -> ...)

    Perhaps the simplest approach here is to buffer all the live descriptions then build the ConfigLoader yaml for processing?
    Alternatively, just provide a config loader yaml file itself for injection into the current cli session.

    NOTE: This is probably completely obsoleted by the `auto` IOpPredicateAutoRunner loader and named ifthen dsl predicate management system.
    """

    name: bool = field(init=False)
    predicate: str = field(init=False)

    def argmap(self):
        return [
            DArg(
                "name",
                desc="Name of this predicate for attaching to other places (can also be a special override command)",
            ),
            DArg(
                "*predicate",
                convert=lambda x: " ".join(x),
                desc="Content of predicate and command to execute upon completion",
            ),
        ]

    async def run(self):
        match self.name:
            case ":load":
                doit = pathlib.Path(self.predicate)
                logger.info("Loading predicate config file: {}", doit)
                match doit.suffix:
                    case ".yaml" | ".yml":
                        loader = ifthen.IfThenConfigLoader(self.state.ifthenRuntime)
                    case ".ifthen":
                        loader = ifthen_dsl.IfThenDSLLoader(self.state.ifthenRuntime)
                    case _:
                        logger.error(
                            "[{}] Filename must end in .yaml, .yml, or .ifthen depending on flow control syntax.",
                            doit,
                        )
                        return

                content = doit.read_text()
                logger.info("[{}] Generate predicates from:\n{}", doit, content)
                created, starts, populate = loader.load(content, activate=False)

                # after loaded/created/activated, we now must POPULATE each predicate with our custom data function attachments
                for pid in populate:
                    await self.state.predicateSetup(self.state.ifthenRuntime[pid])

                loader.activate(starts)

                logger.info("[{}] Created {} predicates!", doit, created)
            case _:
                logger.info("[{}] -> {}", self.name, self.predicate)


@dataclass
class IOpPredicateAutoRunner(IOp):
    """Create/enable or stop named predicate DSL configurations.

    Purpose: Populate content of pre-existing predicate complate configs with live configuration
             data then run/enable the predicate to operate using live market data and live account access.

    Implementation note: This is a refactored/second-generation version of the `ifgroup :load` system where, instead of only loading
                         from static files, the files can be templates which we live-populate at runtime (and the system can have some
                         built-in templates or operate from self-assembled template strings directly instead of needing everything to
                         be from actual files).

    Usage like:
      auto start builtin:algo_flipper.dsl spy-flipper watch_symbol=SPY algo_symbol=SPY evict_symbol=SPY trade_symbol=SPY qty=500 profit_pts=2 loss_pts=2
      auto stop spy-long
      auto report spy-long

    Basically we need to:
      - create predicate config from IfThenRuntimeTemplateExecutor
      - populate user-provided variables/config into the predicate (symbol details, algo, duration, qty, profit/loss targets)
      - start the predicate
      - when trades execute, potentially update the template executor state for success/failure reporting (requires more structure around _what_ is triggering each order and mapping order ids back to "order originiating owners")
    """

    action: str = field(init=False)
    templateName: str = field(init=False)
    name: str = field(init=False)
    properties: dict[str, str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "action",
                convert=lambda x: x.lower(),
                verify=lambda x: x
                in {"start", "stop", "preview", "report-all", "report"},
                desc="Command for predicate runner. one of: start, stop, report",
            ),
            DArg(
                "templateName",
                verify=lambda x: ":" in x
                and x.split(":")[0].lower() in {"builtin", "file"},
                desc="Name of template to pull for applying replacements and creating predicates. Note: template format is kind:name where kind is `builtin` or `file`",
            ),
            DArg(
                "name",
                desc="Name for template+parameters combination (scoped to `templateName` for all operations)",
            ),
            DArg(
                "*properties",
                # *properties format is: ["algo_symbol=SPY", "qty=50", ...]
                # and we want to convert it to a key-val dict, so... split on each '=' then take k/v pairs
                convert=lambda x: dict([kv.strip().split("=", 1) for kv in x]),
                desc="key-value parameters used in template",
            ),
        ]

    def parseProperties(self):
        """Replace any positional markers like :N in param values with the contract names."""
        for k, v in self.properties.items():
            if ":" in v:
                # capture :N values
                # resolve :N values to underlying symbol name
                # replace resolved value in original string for each occurrance
                self.properties[k] = self.state.scanStringReplacePositionsWithSymbols(v)

        return self.properties

    async def run(self):
        it = self.state.ifthenTemplates

        def loadTemplate():
            # Template names must be namespaced so we can look them up correctly.
            # Either: builtin:name or file:name
            frm, tn = self.templateName.lower().split(":")

            # replace any ":N" references in the template parameters with symbol names (where useful)
            self.parseProperties()

            # Running the creation is idempotent if the content between names and targets are the same between calls.
            match frm:
                # Note: 'templateName' is the, well, template name, but each _instance_ of a template is self.name
                case "builtin":
                    it.from_builtin(tn, self.templateName)
                case "file":
                    it.from_file(tn, self.templateName)
                case _:
                    raise ValueError(
                        f"templateName arg must have namespace builtin: or file: - given {self.templateName}"
                    )

        match self.action:
            case "preview":
                loadTemplate()
                t = it.preview_template(self.templateName, self.properties)
                logger.info("[{}] {}", self.templateName, t)
            case "start":
                loadTemplate()
                # populate template with name and args
                # TODO: debug why we need enable=True here _and_ manual .activate() at the end,
                #       since 'enable=True' here should cause the runtime to genreate an activation itself?
                #       We lost track of how the activations work somewhere along the way and which ones
                #       are enabling which features.
                created_count, start_ids, all_ids = it.activate(
                    self.templateName, self.name, self.properties, enable=True
                )

                # this is just a silly way of telling mypy we do not have any `None` entries in
                # the resolved predicates we are about to call .predicateSetup() on for all entries.
                crs = filter(None, [self.state.ifthenRuntime[pid] for pid in all_ids])

                # attach data handlers to all individual predicate ids needing data updates
                await asyncio.gather(*[self.state.predicateSetup(c) for c in crs])

                # start the top-level predicate entry points
                for start_id in start_ids:
                    logger.info("Starting ifthen predicate id: {}", start_id)
                    self.state.ifthenRuntime.activate(start_id)
            case "stop":
                it.deactivate(self.templateName, self.name)
            case "report-all":
                logger.info(
                    "{}",
                    pp.pformat(it.get_system_health_report()),
                )
            case "report":
                logger.info(
                    "[{} :: {}] :: {}",
                    self.templateName,
                    self.name,
                    pp.pformat(
                        it.get_performance_summary(self.templateName, self.name)
                    ),
                )
                logger.info(
                    "[{} :: {}] :: {}",
                    self.templateName,
                    self.name,
                    pp.pformat(it.get_performance_events(self.templateName, self.name)),
                )
            case _:
                logger.error(
                    "[{} :: {}] :: Unknown command?", self.templateName, self.name
                )


@dataclass
class IOpPredicateClearAll(IOp):
    """Delete ALL predicates immediately."""

    everything: bool = field(init=False)

    def argmap(self):
        return [
            DArg(
                "everything",
                default=False,
                desc="By default, only clear active predicates. If 'everything' is True, clear ALL cached predicates for all symbols",
            )
        ]

    async def run(self):
        if self.everything:
            logger.info("Deleting ALL Predicates")
            self.state.ifthenRuntime.clear()
        else:
            logger.info(
                "Stopping all predicates, but they still exist to be re-activated"
            )
            self.state.ifthenRuntime.clearActive()


# ===========================================================================
# COMMAND MAP STUFF
# ===========================================================================

# TODO: potentially split these out into indepdent plugin files?
OP_MAP: Mapping[str, Mapping[str, Type[IOp]]] = {
    "Live Market Quotes": {
        "qquote": IOpQQuote,
        "depth": IOpDepth,
        "add": IOpQuotesAdd,
        "align": IOpQuotesAlign,
        "oadd": IOpQuotesAddFromOrderId,
        "remove": IOpQuotesRemove,
        "rm": IOpQuotesRemove,
        "chains": IOpOptionChain,
        "prequalify": IOpPreQualify,
        "range": IOpOptionChainRange,
    },
    "Order Management": {
        "limit": IOpOrderLimit,
        "buy": IOpOrder,
        "scale": IOpScaleOrder,
        "fast": IOpOrderFast,
        "modify": IOpOrderModify,
        "evict": IOpPositionEvict,
        "cancel": IOpOrderCancel,
        "straddle": IOpStraddleQuote,
    },
    "Predicate Management": {
        "ifthen": IOpPredicateCreate,
        "iflist": IOpPredicateList,
        "ifls": IOpPredicateList,
        "ifrm": IOpPredicateDelete,
        "ifclear": IOpPredicateClearAll,
        "ifgroup": IOpPredicateGroup,
        "auto": IOpPredicateAutoRunner,
    },
    "Portfolio": {
        "balance": IOpBalance,
        "positions": IOpPositions,
        "ls": IOpPositions,
        "orders": IOpOrders,
        "executions": IOpExecutions,
        "report": IOpOrderReport,
    },
    "Connection": {
        "rid": IOpRID,
    },
    "Utilities": {
        "cash": IOpCash,
        "alias": IOpAlias,
        "calendar": IOpCalendar,
        "math": IOpCalculator,
        "info": IOpInfo,
        "details": IOpDetails,
        "expand": IOpExpand,
        "set": IOpSetEnvironment,
        "unset": IOpUnSetEnvironment,
        "say": IOpSay,
        "clear": IOpClearDetails,
        "qualify": IOpQualify,
        "simulate": IOpSimulate,
        "paper": IOpPaper,
        "reconnect": IOpReconnect,
        "daydumper": IOpDayDumper,
        "alert": IOpAlert,
        "advice": IOpAdviceMode,
        "reporter": IOpMarketReporter,
    },
    "Schedule Management": {
        # full "named" versions of the commands
        "sched-add": IOpScheduleEvent,
        "sched-list": IOpScheduleEventList,
        "sched-cancel": IOpScheduleEventCancel,
        # also allow short versions of the commands
        "sadd": IOpScheduleEvent,
        "slist": IOpScheduleEventList,
        "scancel": IOpScheduleEventCancel,
    },
    "Task Management": {
        "tasklist": IOpTaskList,
        "taskcancel": IOpTaskCancel,
    },
    "Quote Management": {
        "qsave": IOpQuoteSave,
        "qadd": IOpQuoteAppend,
        "qremove": IOpQuoteRemove,
        "qrm": IOpQuoteRemove,
        "qdelete": IOpQuoteGroupDelete,
        "qrestore": IOpQuoteRestore,
        "qclean": IOpQuoteClean,
        "qlist": IOpQuoteList,
        "qsnapshot": IOpQuoteSaveClientSnapshot,
        "qloadsnapshot": IOpQuoteLoadSnapshot,
        "colorset": IOpColors,
        "colorsload": IOpColorsLoad,
    },
}


@dataclass
class Dispatch:
    def __post_init__(self):
        self.d = mutil.dispatch.Dispatch(OP_MAP)

    def runop(self, *args, **kwargs) -> Coroutine:
        return self.d.runop(*args, **kwargs)
