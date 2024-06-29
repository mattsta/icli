import dataclasses  # just for .replace
from dataclasses import dataclass, field
from typing import *
import bisect
import calendar
import datetime
import enum

import fnmatch
import itertools
import math
import pathlib

import sys
from collections import Counter, defaultdict

import mutil.dispatch
import mutil.expand
import numpy as np

import pandas as pd

from ib_async import Bag, Contract, Order
from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame
from mutil.numeric import fmtPrice
from mutil.timer import Timer
from icli.helpers import *
import asyncio

import aiohttp

import pendulum

import prettyprinter as pp
import tradeapis.buylang as buylang
from questionary import Choice

import icli.orders as orders
# from .accum import AccumulatorAgent, AccumulatorController

# from .agent import AgentController, AgentSymbol

pp.install_extras(["dataclasses"], warn_on_error=False)

# TODO: convert to proper type and find all misplaced uses of "str" where we want Symbol.
# TODO: also break out Symbol vs LocalSymbol usage
Symbol = str

OrderExtension = enum.Enum("OrderExtension", "PROFIT LOSS BRACKET NONE")

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

# Also map for user typing shorthand on command line order entry.
# Values abbreviations are allowed for easier command typing support.
# NOTE: THIS LIST IS USED TO TRIGGER ORDERS IN order.py:IOrder().order() so THESE NAMES MUST MATCH
#       THE OFFICIAL IBKR ALGO NAME MAPPINGS THERE.
# This is a TRANSLATION TABLE between our "nice" names like 'AF' and the IBKR ALGO NAMES USED FOR ORDER PLACING.
# NOTE: DO NOT SEND IOrder.order() requests using 'AF' because it must be LOOKED UP HERE FIRST.
# TODO: on startup, we should assert each of these algo names match an actual implemented algo order method in IOrder().order()
ALGOMAP = dict(
    LMT="LMT",
    LIM="LMT",
    LIMIT="LMT",
    AF="LMT + ADAPTIVE + FAST",
    AS="LMT + ADAPTIVE + SLOW",
    MID="MIDPRICE",
    MIDPRICE="MIDPRICE",
    MTL="MTL",  # MARKET-TO-LIMIT (execute at top-of-book, but don't sweep, just set a limit for remainder)
    PRTMKT="MKT PRT",  # MARKET-PROTECT (futs only), triggers immediately
    PRTSTOP="STOP PRT",  # STOP WITH PROTECTION (futs only), triggers when price hits
    PEGMID="PEG MID",  # Floating midpoint peg, must be directed IBKRATS or IBUSOPT
    REL="REL",
    STOP="STP",
    STP="STP",
    MKT="MKT",
    MIT="MIT",
    LIT="LIT",
    AFM="MKT + ADAPTIVE + FAST",
    AMF="MKT + ADAPTIVE + FAST",
    ASM="MKT + ADAPTIVE + SLOW",
    AMS="MKT + ADAPTIVE + SLOW",
    MOO="MOO",
    MOC="MOC",
)

# This is a little redundant since it's also configured in cli.py.
# We could potentially just move this to helpers.py to fix the import problem and have
# a single source of truth for the number here.
ICLI_CLIENT_ID = int(os.getenv("ICLI_CLIENT_ID", 0))


def addRowSafe(df, name, val):
    """Weird helper to stop a pandas warning.

    Fixes this warning pandas apparently is now making for no reason:
    FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
                   In a future version, this will no longer exclude empty or all-NA columns when determining
                   the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation."""
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
        limit = round(price * EQUITY_BOUNDS, 2)

        if isinstance(contract, (Option, FuturesOption)):
            # options have deeper exit floor criteria because their ranges can be wider.
            # the IBKR algo is "adaptive fast" so it should *try* to pick a good value in
            # the spread without immediately going to market, but ymmv.
            limit = round(price * OPTION_BOUNDS, 2)

            # if price is too small, it may be moving faster, so increase the limit slightly more
            # (this is still always bound by the market spread anyway; we're basically doing an
            #  excessively high effort market order but just trying to protect against catastrophic fills)
            if limit < 2:
                limit = round(limit * OPTIONS_BOOST, 2)
    else:
        # else, position IS SELLING, we want to chase LOWER PRICES to GET THE ORDER
        limit = round(price / EQUITY_BOUNDS, 2)

        if isinstance(contract, (Option, FuturesOption)):
            # options have deeper exit floor criteria because their ranges can be wider.
            # the IBKR algo is "adaptive fast" so it should *try* to pick a good value in
            # the spread without immediately going to market, but ymmv.
            limit = round(price / OPTION_BOUNDS, 2)

            # (see logic/rationale above)
            if limit < 2:
                limit = round(limit / OPTIONS_BOOST, 2)

    if isinstance(contract, (Option, FuturesOption, Future)):
        limit = comply(contract, limit)

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


@dataclass
class IOpQQuote(IOp):
    """Quick Quote: Run a temporary quote request then print results when volatility is populated."""

    def argmap(self) -> list[DArg]:
        return [DArg("*symbols")]

    async def run(self):
        if not self.symbols:
            logger.error("No symbols requested?")
            return

        contracts = [contractForName(sym) for sym in self.symbols]

        await self.state.qualify(*contracts)

        if not all(c.conId for c in contracts):
            logger.error("Not all contracts reported successful lookup!")
            logger.error(contracts)
            return

        # BROKEN IBKR WORKAROUND:
        # For IV/HV calculations, IBKR seems to have a server-side background
        # process to generate them on-demand, but if they don't exist
        # when your quote is requested, the fields will never be
        # populated.
        # Requesting quotes outside of RTH often takes 5-30 seconds to
        # deliver IV/HV calculations (if they get populated at all).
        # So we have to REQUEST the data, hope IBKR spawns its
        # background calculations, CANCEL the request, then RE-REQUEST
        # the data and hopefully it is populated the second time.
        # Also, the first cancel must be delayed enough to trigger their
        # background calculations or else they still won't populate
        # in the future.
        # Also also, the second request must be soon after the first
        # (but not *TOO* soon!) or else their background cache will
        # clear their background-created cached value and the entire
        # double-request cycle needs to start again.
        totalTry = 0
        ATTEMPT_LIMIT = 10
        while True:
            tickers = []
            logger.info(
                "Requesting tickers for {}",
                ", ".join(
                    [c.localSymbol.replace(" ", "") or c.symbol for c in contracts]
                ),
            )
            for contract in contracts:
                # Request quotes with metadata fields populated
                # (note: metadata is only populated using "live" endpoints,
                #  so we can't use the self-canceling "11 second snapshot" parameter)
                tickers.append(
                    self.ib.reqMktData(
                        contract,
                        tickFieldsForContract(contract),
                    )
                )

            # Loop over quote results until they have all been reported
            success = False
            for i in range(0, 3):
                ivhv = [(t.impliedVolatility, t.histVolatility) for t in tickers]

                # if any iv/hv are still nan, don't stop yet.
                if np.isnan(ivhv).any() and totalTry < ATTEMPT_LIMIT:
                    logger.warning(
                        "Waiting for data to arrive... (attempt {} of {})",
                        totalTry,
                        ATTEMPT_LIMIT,
                    )
                    totalTry += 1
                else:
                    if totalTry >= ATTEMPT_LIMIT:
                        logger.warning("Not all quotes finished fetching. Final state:")

                    # if all iv and hv are populated, stop!
                    success = True

                    df = pd.DataFrame(tickers)

                    # extract contract data from nested object pandas would otherwise
                    # just convert to a blob of json text.
                    contractframe = pd.DataFrame([t.contract for t in tickers])

                    if contractframe.empty:
                        logger.error("No result!")
                        continue

                    contractframe = contractframe["symbol secType conId".split()]

                    # NB: 'halted' statuses are:
                    # -1 Halted status not available.
                    # 0 Not halted.
                    # 1 General halt. regulatory reasons.
                    # 2 Volatility halt.
                    df = df[
                        """bid bidSize
                           ask askSize
                           last lastSize
                           volume open high low close
                           halted shortableShares
                           histVolatility impliedVolatility""".split()
                    ]

                    # attach inner name data to data rows since it's a nested field thing
                    # this 'concat' works because the row index ids match across the contracts
                    # and the regular ticks we extracted.
                    df = pd.concat([contractframe, df], axis=1)

                    printFrame(df)
                    break

                await asyncio.sleep(2)

            if success:
                break

            for contract in contracts:
                self.ib.cancelMktData(contract)

            # try again...
            await asyncio.sleep(0.333)

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

    def argmap(self):
        return [DArg("key"), DArg("*val")]

    async def run(self):
        # forward map of keys -> shorthand aliases for the key
        KEY_ALIASES = dict(exchange=set("x xch xchange".split()))

        # reverse map of each unique alias -> original key
        ALIASES_KEYS = {vx: k for k, v in KEY_ALIASES.items() for vx in v}

        key = self.key.lower()
        val = self.val[0] if self.val else None

        # attempt to use shorthand for key, but if no shorthand, use key directly anyway
        key = ALIASES_KEYS.get(key, key)

        self.state.updateGlobalStateVariable(key, val)


@dataclass
class IOpSay(IOp):
    """Speak a custom phrase provided as arguments.

    Can be used as a standalone command or combined with scheduled events to create
    speakable events on a delay."""

    def argmap(self):
        return [DArg("*what")]

    async def run(self):
        asyncio.create_task(self.state.speak.say(say=" ".join(self.what)))


@dataclass
class IOpPositionEvict(IOp):
    """Evict a position using automatic MIDPRICE sell order for equity or ADAPTIVE FAST for options and futures.

    Note: the symbol name accepts '*' for wildcards!

    Also note: for futures, the actual symbol has the month expiration attached like "MESU2", so the portfolio
               symbol is not just "MES". Evicting futures reliably uses evict MES* and not MES or /MES.
    """

    def argmap(self):
        return [
            DArg("sym"),
            DArg(
                "qty",
                convert=float,
                verify=lambda x: x != 0 and x >= -1,
                desc="qty is the exact quantity to evict (or -1 to evict entire position)",
            ),
            DArg(
                "delta",
                convert=float,
                verify=lambda x: 0 <= x <= 1,
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
            price = (bid + ask) / 2

            if self.delta:
                # verify quote is loaded...
                if not self.state.quoteExists(contract):
                    logger.info("Quote didn't exist, adding now...")
                    await self.runoplive(
                        "add",
                        f'"{quotesym}"',
                    )

                # if asking for a delta eviction, check current quote...
                quotesym = lookupKey(contract)

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
                await self.state.qualify(contract)

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
class IOpAlert(IOp):
    """Enable or disable trade completion sound effects."""

    def argmap(self):
        return [
            DArg(
                "cmd",
                convert=lambda x: x.lower(),
                verify=lambda x: x in {"yes", "no", "status", "on", "off"},
                desc="Yes or No or Status to enable/disable/view trade sound effect state.",
            )
        ]

    async def run(self):
        if self.cmd in {"yes", "on"}:
            logger.info("Setting Alerts ON!")
            self.state.alert = True
        elif self.cmd in {"no", "off"}:
            logger.info("Setting Alerts OFF!")
            self.state.alert = False
        else:
            logger.info("Alert state: {}", self.state.alert)


@dataclass
class IOpCalendar(IOp):
    """Just show a calendar!"""

    def argmap(self):
        return [
            DArg(
                "*year",
                desc="Year for your calendar to show (if not provided, just use current year)",
            )
        ]

    async def run(self):
        try:
            year = int(self.year[0])
        except:
            year = pendulum.now().year

        # MURICA
        # (also lol for this outdated python API where you have to globally set the calendar start
        #  date for your entire environment!)
        calendar.setfirstweekday(calendar.SUNDAY)
        logger.info("[{}] Calendar:\n{}", year, calendar.calendar(year, 1, 1, 6, 3))


@dataclass
class IOpCalculator(IOp):
    """Just show a calculator!"""

    def argmap(self):
        return [DArg("*parts", desc="Calculator input")]

    async def run(self):
        cmd = " ".join(self.parts)

        try:
            logger.info("[{}]: {:,.4f}", cmd, self.state.calc.calc(cmd))
        except Exception as e:
            logger.warning("[{}]: calculation error: {}!", cmd, e)


@dataclass
class IOpInfo(IOp):
    """Show the underlying IBKR contract object for a symbol.

    This is mainly useful to verify the IBKR details or extract underlying contract IDs
    for other debugging or one-off usage purposes."""

    def argmap(self):
        return [DArg("*symbols", desc="Symbols to check for contract details")]

    async def run(self):
        contracts = []

        for sym in self.symbols:
            # yet another in-line hack for :N lookups because we still haven't created a central abstraction yet...
            if sym[0] == ":":
                contracts.append(self.state.quoteResolve(sym)[1])
            else:
                contracts.append(contractForName(sym))

        await self.state.qualify(*contracts)
        logger.info("{}", pp.pformat(contracts))


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

    def argmap(self):
        return [
            DArg(
                "*parts", desc="Command to expand then execute all combinations thereof"
            )
        ]

    async def run(self):
        # build each part needing expansion
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
        await asyncio.gather(*[self.runoplive(c[0], c[1]) for c in cmds])


@dataclass
class IOpScheduleEvent(IOp):
    """Schedule a command to execute at a specific date+time in the future."""

    # asub /NQ COMBONQ yes 0.66 cash 15 TemaTHMAFasterSlower direct
    def argmap(self):
        return [
            DArg(
                "name",
                desc="Name of event (for listing and canceling in the future if needed)",
            ),
            DArg(
                "datetime",
                convert=lambda dt: pendulum.parse(dt, tz="US/Eastern"),
                desc="Date and Time of event (timezone will be Eastern Time)",
            ),
            DArg("*cmd", desc="icli command to run at the given time"),
        ]

    async def run(self):
        if self.name in self.state.scheduler:
            logger.error(
                "[{} :: {}] Can't schedule because name already scheduled!",
                self.name,
                self.cmd,
            )
            return False

        now = pendulum.now()

        # "- 1 second" allows us to schedule for "now" without time slipping into the past and
        # complaining we scheduled into the past. sometimes we just want it now.
        if (now - pendulum.duration(seconds=1)) > self.datetime:
            logger.error(
                "You requested to schedule something in the past? Not scheduling."
            )
            return False

        logger.info(
            "[{} :: {} :: {}] Scheduling: {}",
            self.name,
            self.datetime,
            (self.datetime - now).in_words(),
            self.cmd,
        )

        async def doit() -> None:
            try:
                howlong = (self.datetime - now).in_seconds()
                logger.info(
                    "[{} :: {}] command is scheduled to run in {:,.2f} seconds ({:,.2f} minutes)!",
                    self.name,
                    self.cmd,
                    howlong,
                    howlong / 60,
                )

                await asyncio.sleep(howlong)

                # "self.cmd" is an array of commands to run...
                for cmd in self.cmd:
                    logger.info("[{} :: {}] RUNNING UR CMD!", self.name, cmd)
                    c, *v = cmd.split(" ", 1)
                    await self.runoplive(c, v[0] if v else None)
                    logger.info("[{} :: {}] Completed UR CMD!", self.name, cmd)
            except asyncio.CancelledError:
                logger.warning(
                    "[{} :: {}] Future Scheduled Task Canceled!", self.name, self.cmd
                )
            except:
                logger.exception(
                    "[{} :: {}] Scheduled event failed?", self.name, self.cmd
                )
            finally:
                del self.state.scheduler[self.name]
                logger.info("[{}] Removed scheduled event!", self.name)

        sched = asyncio.create_task(doit())

        # save reference so this task doesn't get GC'd (also so we can cancel it manually too)
        self.state.scheduler[self.name] = (self.datetime, self.cmd, sched)
        logger.info("[{} :: {}] Scheduled via: {}", self.name, self.cmd, sched)


@dataclass
class IOpScheduleEventList(IOp):
    """List scheduled events by name and command and target date."""

    async def run(self):
        logger.info("Listing {} scheduled events by name...", len(self.state.scheduler))
        for name, (when, cmd, task) in sorted(
            self.state.scheduler.items(), key=lambda i: i[1][0]
        ):
            logger.info("[{} :: {}] {} ({})", name, when, cmd, task)


@dataclass
class IOpScheduleEventCancel(IOp):
    """Cancel event by name."""

    def argmap(self):
        return [DArg("name", desc="Name of event to cancel")]

    async def run(self):
        got = self.state.scheduler.get(self.name)
        if not got:
            logger.error("[{}] Scheduled event not found?", self.name)
            return False

        when, cmd, task = got
        logger.info("[{} :: {}] Cancelling scheduled task!", self.name, cmd)

        # the .cancel() thows an exception in the task which deletes
        # itself from the scheduler, so DO NOT 'del' here.
        task.cancel()

        logger.info("[{} :: {}] Task deleted!", self.name, cmd)


@dataclass
class IOpAlias(IOp):
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

        cmd: dict[str, dict[str, list[str]]] = aliases[self.cmd]
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
        try:
            if self.sym.startswith(":"):
                self.sym, contract = self.state.quoteResolve(self.sym)
                assert self.sym
            else:
                contract = contractForName(self.sym)
                await self.state.qualify(contract)

            assert contract.localSymbol
        except:
            logger.error("No contract found for: {}", self.sym)
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
                    "[{}] Depth not populated. Failing warm-up check {}.",
                    contract.localSymbol,
                    i,
                )

                if i > 20:
                    logger.error("Depth not populated in expected time!")
                    return

                await asyncio.sleep(0.15)

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

            def decimal_formatter(x):
                # float(x) to enforce at least one decimal place into existence if this was an integer
                str_x = str(float(x))
                parts = str_x.split(".")

                assert (
                    len(parts) == 2
                ), "Why don't you have a decimal number here? What else could this be?"

                after_decimal_str = parts[1]
                decimal_size = len(after_decimal_str)

                # always print even number of decimals if we reach here
                # (i.e. don't print $123.0 next to $123.25 or $1.234 next to $1.2345)
                if decimal_size % 2 != 0:
                    decimal_size += 1

                return f"{x:,.{decimal_size}f}"

            # condition dataframe reorganization on the input list existing.
            # for some smaller symbols, bids or asks may not get returned
            # by the flaky ibkr depth APIs
            if t.domBids:
                adb = pd.DataFrame(t.domBids)
                fixedBids = (
                    adb.groupby("price", as_index=False)
                    .agg({"size": "sum", "marketMaker": list})
                    .convert_dtypes()
                    .sort_values(by=["price"], ascending=False)
                    .reset_index(drop=True)
                )

                # format floats as currency strings with proper cent padding.
                fixedBids["price"] = fixedBids["price"].apply(decimal_formatter)
            else:
                fixedBids = pd.DataFrame([dict(size=0)])

            if t.domAsks:
                adf = pd.DataFrame(t.domAsks)
                # logger.info("Asks: {}", adf.to_string())
                fixedAsks = (
                    adf.groupby("price", as_index=False)
                    .agg({"size": "sum", "marketMaker": list})
                    .convert_dtypes()
                    .sort_values(by=["price"], ascending=True)
                    .reset_index(drop=True)
                )

                fixedAsks["price"] = fixedAsks["price"].apply(decimal_formatter)
            else:
                fixedAsks = pd.DataFrame([dict(size=0)])

            # generate a synthetic sum row then add commas to the sums after summing.......
            fixedBids.loc["sum", "size"] = fixedBids["size"].sum()
            fixedBids["size"] = fixedBids["size"].apply(lambda x: f"{x:,}")

            fixedAsks.loc["sum", "size"] = fixedAsks["size"].sum()
            fixedAsks["size"] = fixedAsks["size"].apply(lambda x: f"{x:,}")

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
        # logger.info("current orderS: {}", pp.pformat(ords))
        promptOrder = [
            Q(
                "Current Order",
                choices=[
                    Choice(
                        f"{o.contract.localSymbol or o.contract.symbol:<21} {o.order.action:<4} {o.order.totalQuantity:<6} {o.order.orderType} {o.order.tif} lmt:${fmtPrice(o.order.lmtPrice):<7} aux:${fmtPrice(o.order.auxPrice):<7}",
                        o,
                    )
                    for o in sorted(
                        filter(
                            lambda x: x.orderStatus.clientId == self.state.clientId,
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
            # TODO: we should be rounding these limit prices too because IBKR complains about incorrect tick compliance on updates.
            if lmt:
                # COPY the underlying order so we don't directly modify the internal trade cache
                # (so if the order update fails to apply, our data remains in a good state)
                ordr = dataclasses.replace(ordr, lmtPrice=float(lmt))

            if stop:
                ordr = dataclasses.replace(ordr, auxPrice=float(stop))

            if qty:
                # TODO: allow quantity of -1 or "ALL" to use total position?
                #       We would need to look up the current portfolio holdings to match sizes against here.
                ordr = dataclasses.replace(ordr, totalQuantity=float(qty))

            if ordr.parentId:
                logger.warning("Removing parent id from updated order!")
                ordr = dataclasses.replace(ordr, parentId=0)

            # we MUST have replaced the order by now or else the conditions above are broken
            assert ordr != trade.order

            # Also, MODIFY must not attempt to populate an algo or else IBKR complains (but only sometimes)
            # (primarily found when using Adaptive aglos (AF/AS) on spreads, it complains "Cannot change to the new order type" even though we didn't
            #  modify the algo type? — or, it looks like the API is configuring Adaptive orders to have orderType IBALGO when
            #  after they are submitted, so when we re-submit with just LMT+Adaptive, IBKR thinks we are "changing" the algo when actually
            #  their API change dit out from under us)
            if ordr.orderType == "IBALGO":
                ordr.orderType = "LMT"

            logger.info("Submitting order update: {} :: {}", contract, ordr)
            trade = self.ib.placeOrder(contract, ordr)
            logger.info("Updated: {}", pp.pformat(trade))
        except KeyboardInterrupt:
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
class IOpOrder(IOp):
    """Quick order entry with full order described on command line."""

    def argmap(self):
        # TODO: write a parser for this language instead of requiring fixed orders for each parameter?
        # allow symbol on command line, optionally
        # BUY IWM TOTAL 50000 ALGO LIMIT/LIM/LMT AF AS REL MP AMF AMS MOO MOC
        # TODO: find a way to make this aware of margin requirements for futures, currencies, etc
        # TODO: improve the dispatch system to allow someting like named arguments so we don't need to inject our own positional name prefixes here.
        #       Would also easily allow us to maybe have xor args where we want to switch between "total cost" and "total quantity" as inputs.
        return [
            DArg("symbol", convert=lambda x: x.upper()),
            DArg("total", convert=lambda x: PriceOrQuantity(x)),
            DArg(
                "algo",
                convert=lambda x: x.upper(),
                verify=lambda x: x in ALGOMAP.keys(),
                errmsg=f"Available algos: {pp.pformat(ALGOMAP)}",
            ),
            DArg("*preview", desc="Print as a what-if order to show margin impact." ""),
        ]

    async def run(self) -> bool:
        contract = None
        if " " in self.symbol:
            # is multi-leg spread/bag request, so do bag
            orderReq = self.state.ol.parse(self.symbol)
            contract = await self.state.bagForSpread(orderReq)
        else:
            # else, is a regular single symbol
            # if ordering current positional quote, do the lookup.
            # TODO: make centralized lookup helper function
            if self.symbol.startswith(":"):
                orig = self.symbol
                self.symbol, contract = self.state.quoteResolve(self.symbol)
                if not self.symbol:
                    logger.error("[{}] Failed to find symbol by position index!", orig)
                    return None
            else:
                contract = contractForName(self.symbol)

        if contract is None:
            logger.error("Not submitting order because contract can't be formatted!")
            return None

        if not isinstance(contract, Bag):
            # spreads are qualified when they are initially populated so we never qualify a bag directly
            await self.state.qualify(contract)

        isLong = self.total.is_long

        # send the order to IBKR
        am = ALGOMAP[self.algo]

        # we're double using 'preview' for actual 'preview' and also an optional '@ $33.22' limit price field,
        # so this is only a true 'preview' if there is only one element here.
        optionalPrice = 0
        isPreview = False
        bracket = None

        if self.preview:
            # preview is just ["preview"] in self.preview
            isPreview = len(self.preview) == 1

            pl = len(self.preview)
            # self.preview can *also* be a list with a limit price where we use syntax of ["@", "33.33"]
            if pl >= 2:
                if self.preview[0] != "@":
                    logger.warning(
                        "Syntax broken? Expected '@ <price>' but got: {}", self.preview
                    )

                optionalPrice = float(self.preview[1])

                # if we have EVEN MORE ELEMENTS, user is requesting an auto-attached exit too.

                extension = OrderExtension.NONE
                extensionPriceProfit = None
                extensionPriceLoss = None
                if pl >= 4:
                    try:
                        extensionPrice = float(self.preview[3])
                    except:
                        logger.error(
                            "Failed to convert exit price: {} from {}",
                            self.preview[3],
                            self.preview,
                        )
                        return

                    # We want to submit attached orders, but IBKR only allows attacahed orders as either:
                    #  - dual side braacket orders
                    #  - trailing stop limit orders

                    # If short, we need to flip the direction of our math and it's easier to just invert here
                    if not isLong:
                        extensionPrice *= -1

                    match self.preview[2]:
                        case "+":
                            # Create PROFIT exit at +$A.BC
                            extension = OrderExtension.PROFIT
                            extensionPriceProfit = optionalPrice + extensionPrice
                            extensionPriceLoss = max(
                                optionalPrice - (extensionPrice * 4), 0.05
                            )
                            bracket = Bracket(profitLimit=extensionPriceProfit)
                        case "-":
                            # Create LOSS exit at -$X.YZ
                            extension = OrderExtension.LOSS
                            extensionPriceLoss = optionalPrice - extensionPrice
                            bracket = Bracket(lossLimit=extensionPriceLoss)
                            extensionPriceProfit = optionalPrice + (extensionPrice * 4)
                        case "±":
                            # Create BRACKET.
                            # BRACKET can be submitted with the order ALL AT ONCE as a conditional parent tree order.
                            extension = OrderExtension.BRACKET
                            extensionPriceLoss = optionalPrice - extensionPrice
                            extensionPriceProfit = optionalPrice + extensionPrice
                            bracket = Bracket(
                                profitLimit=extensionPriceProfit,
                                lossLimit=extensionPriceLoss,
                            )
                        case _:
                            logger.error(
                                "Invalid order exit requested? Expected one of +|-|± but got {}",
                                self.preview[2],
                            )
                            return

                # we can *also* optionally append "preview" to the final price for a preview-at-price order like:
                # buy AAPL 100 AF @ 200.21 preview
                # (yeah, it's a mess, but it's _our_ mess)
                if self.preview[-1].lower().startswith("p"):
                    isPreview = True

        # note: limit=0 causes placeOrderForContract to automatically calculate the quote midpoint as starting offer.
        placed = await self.state.placeOrderForContract(
            self.symbol,
            isLong,
            contract,
            qty=self.total,
            # if we are buying by AMOUNT, we don't specify a limit price since it will be calculated automatically,
            # then we read the calculated amount to run the auto-price-tracking attempts.
            # if we provide a price, it gets used as the limit price directly.
            limit=optionalPrice,
            orderType=am,
            preview=isPreview,
            bracket=bracket,
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
        if optionalPrice:
            logger.warning(
                "Not running price algo because limit price provided directly..."
            )
            return False

        order, trade = placed

        quoteKey = lookupKey(contract)

        checkedTimes = 0
        # while (unfilled quantity) AND (order NOT canceled or rejected or broken)
        while (rem := trade.orderStatus.remaining) > 0 or (
            "Pending" in trade.orderStatus.status
        ):
            if ("Cancel" in trade.orderStatus.status) or (
                trade.orderStatus.status in {"Filled", "Inactive"}
            ):
                logger.error(
                    "[{} :: {}] Order was canceled or rejected! Status: {}",
                    trade.orderStatus.status,
                    trade.contract.localSymbol,
                    pp.pformat(trade.orderStatus),
                )
                return False

            if rem == 0:
                logger.warning(
                    "Quantity Remaining is zero, but status is Pending. Waiting for final order status update..."
                )
                # sleep 75 ms and check again
                await asyncio.sleep(0.075)
                # TODO: infinite looped here? WHAT?
                continue

            logger.info("Quantity remaining: {:,.2f}", rem)
            checkedTimes += 1

            # if this is the first check after the order was placed, don't
            # run the algo (i.e. give the original limit price a chance to work)
            if checkedTimes == 1:
                logger.info("Skipping adjust so original limit has a chance to fill...")
                await asyncio.sleep(0.075)
                continue

            # get current qty/value of trade both remaining and already executed
            # TODO: fix these names. these names are bad. "currentQty" is actually "order remaining quantity"
            (
                remainingAmount,  # PRICE * Q
                totalAmount,  # PRICE * Q
                currentPrice,  # SYMBOL PRICE
                currentQty,  # REMAINING QUANTITY
            ) = self.state.amountForTrade(trade)

            # re-read current quote for symbol before each update in case the price is moving rapidly against us
            bid, ask = self.state.currentQuote(quoteKey)

            if bid > 0 and ask > 0:
                logger.info("Adjusting price for more aggressive fills...")

                # TODO: restore ability for OPENING ORDERS to reduce quantity when ordering and chasing quotes, but
                #       remember to DO NOT reduce quantity when chasing closing orders because we want to close a whole position.
                if isLong:
                    # if this is a BUY LONG order, we want to close the gap between our initial price guess and the actual ask.
                    newPrice = automaticLimitBuffer(
                        trade.contract, isLong, (currentPrice + ask) / 2
                    )
                else:
                    # else, this is a SELL SHORT (or close long) order, so we want to start at our price and chase the bid closer.
                    newPrice = automaticLimitBuffer(
                        trade.contract, isLong, (currentPrice + bid) / 2
                    )

                logger.info(
                    "Price changing from {} to {} (difference {}) for spending ${:,.2f}",
                    currentPrice,
                    newPrice,
                    round((newPrice - currentPrice), 4),
                    totalAmount,
                )

                if currentPrice == newPrice:
                    logger.error(
                        "Not submitting order update because no price change? Order still active!"
                    )
                    return False

                logger.info("Submitting order update to: ${:,.2f}", newPrice)
                order.lmtPrice = newPrice

                self.ib.placeOrder(contract, order)

            waitDuration = 3
            logger.info(
                "[{} :: {}] Waiting for {} seconds to check for new executions...",
                trade.orderStatus.orderId,
                checkedTimes,
                waitDuration,
            )

            try:
                await asyncio.sleep(waitDuration)
            except:
                # catches CTRL-C during sleep
                logger.warning(
                    "[{}] User canceled automated limit updates! Order still live.",
                    trade.orderStatus.orderId,
                )
                break

        # now just print current holdings so we have a clean view of what we just transacted
        # (but schedule it for a next run so so the event loop has a chance to update holdings first)
        async def delayShowPositions():
            await asyncio.sleep(0.9626)
            await self.runoplive("positions", [])

        asyncio.create_task(delayShowPositions())

        # "return True" signals to an algo caller the order is finalized without errors.
        return True



@dataclass
class IOpOrderStraddle(IOp):
    """Place a long straddle spread order (optionally a strangle) by providing symbol and width in strikes from ATM for put/call strikes (width 0 is just an ATM(ish) straddle)."""

    def argmap(self):
        return [
            DArg(
                "symbol",
                convert=lambda x: x.upper(),
                verify=lambda x: " " not in x,
                desc="Underlying for contracts to use for ATM quote and leg choices",
            ),
            DArg(
                "tradeSymbol",
                convert=lambda x: x.upper(),
                verify=lambda x: " " not in x,
                desc="Symbol to use for order placement. Common usage is by using SPX for underlying but SPXW as the trade symbol.",
            ),
            DArg(
                "amount",
                convert=lambda x: float(x.replace("_", "").replace(",", "")),
                desc="Maximum dollar amount to spend",
            ),
            DArg(
                "width",
                convert=int,
                desc="How many strikes apart to place your legs (can be 0 to just to strangle P/C ATM)",
            ),
            DArg(
                "*preview",
                desc="Run all the calculations, but do not place an active order.",
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
        if self.symbol[0] == ":":
            self.symbol, contract = self.state.quoteResolve(self.symbol)
            assert self.symbol
        else:
            logger.info("[{}] Looking up contract...", self.symbol)

            # we need to hack the symbol lookup if these are index options...
            prefix = "I:" if self.symbol in {"SPX", "NQ", "VX"} else ""
            contract = contractForName(f"{prefix}{self.symbol}")

            await self.state.qualify(contract)
            logger.info("[{}] Found contract: {}", self.symbol, contract)

        if not isinstance(contract, (Stock, Future, Index)):
            logger.error(
                "Spreads are only supported on stocks and futures and index options, but you tried to run a spread on: {}",
                contract,
            )
            return

        # also hack here for index contracts having 'I:' prefix but we don't want it for the quote lookup.
        quotesym = lookupKey(contract)

        quotesym = self.symbol
        quote = self.state.quoteState[quotesym]
        strikes = await self.runoplive("chains", quotesym)

        strikes = strikes[quotesym]

        if not strikes:
            logger.error("[{} :: {}] No strikes found?", self.symbol, quotesym)
            return None

        currentPrice = quote.last
        logger.info("[{}] Using ATM price: {:,.2f}", self.symbol, currentPrice)

        sstrikes = sorted(strikes.keys())

        now = pendulum.now().in_tz("US/Eastern")

        # if after market close, use next day
        if (now.hour, now.minute) >= (16, 15):
            now = now.add(days=1)

        # Note: IBKR Expiration formats in the dict are *full* YYYYMMDD, not OCC YYMMDD.
        expTry = now.date()
        expirationFmt = f"{expTry.year}{expTry.month:02}{expTry.day:02}"
        useExpIdx = bisect.bisect_left(sstrikes, expirationFmt)

        # TODO: add param for expiration days away or keep fixed?
        self.expirationAway = 0
        useExp = sstrikes[useExpIdx : useExpIdx + 1 + self.expirationAway][-1]

        assert useExp in strikes
        useChain = strikes[useExp]

        # nearest ATM strike to the current quote
        currentStrikeIdx = find_nearest(useChain, currentPrice)

        # Note: we are indexing these by STRIKE WIDTH and NOT BY PRICE. We would bisect if using price.
        # LOWER PUT
        putIdx = currentStrikeIdx - self.width

        # HIGHER CALL
        callIdx = currentStrikeIdx + self.width

        put = useChain[putIdx]
        atm = useChain[currentStrikeIdx]
        call = useChain[callIdx]

        logger.info(
            "[{} :: {}] USING SPREAD: [put {:.2f}] <-(${:,.2f} wide)-> [atm {:.2f}] <-(${:,.2f} wide)-> [call {:.2f}]",
            self.symbol,
            self.tradeSymbol,
            put,
            atm - put,
            atm,
            call - atm,
            call,
        )

        assert put <= atm <= call

        spread = " ".join(
            [
                f"{what} 1 {self.tradeSymbol}{useExp[2:]}{direction}{int(strike * 1000):08}"
                for what, direction, strike in {
                    ("buy", "P", put),
                    ("buy", "C", call),
                }
            ]
        )

        closingSpread = " ".join(
            [
                f"{what} 1 {self.tradeSymbol}{useExp[2:]}{direction}{int(strike * 1000):08}"
                for what, direction, strike in {
                    ("sell", "P", put),
                    ("sell", "C", call),
                }
            ]
        )

        logger.info("[{} :: {}] OPEN CHAIN:  {}", self.symbol, self.tradeSymbol, spread)
        logger.info(
            "[{} :: {}] CLOSE CHAIN: {}",
            self.symbol,
            self.tradeSymbol,
            closingSpread,
        )

        orderReq = self.state.ol.parse(spread)

        # subscribe to quote for spread so we can figure out the current price for ordering...
        qk = await self.runoplive("add", f'"{spread}"')

        # add closing spread async because it was causing maybe rate limit problems with IBKR data feeds?
        asyncio.create_task(self.runoplive("add", f'"{closingSpread}"'))

        # logger.info("Got quote keys: {}", qk)

        # find quote key for the spread (not the legs)
        # (kinda weird, but the 'add' returns 3 or 4 keys: one for each leg symbol and one for the combined spread quote to
        #  buy and (maybe) ANOTHER for the combined spread quote to sell, so we want to ONLY extract the BUY SPREAD quote
        #  for this straddle purchase. the spread quote is a tuple key while the leg quotes are just string keys)
        # The Bag BUY quote keys look like: ((640377328, 1, 'BUY', 'SMART', 0, 0, '', -1), (649711537, 1, 'BUY', 'SMART', 0, 0, '', -1))
        quoteKey = next(filter(lambda x: isinstance(x, tuple) and x[0][2] == "BUY", qk))

        # ACTUALLY, the IBKR spread quotes are VERY UNRELIABLE and often just NEVER POPULATE, so build our own spread quotes... sigh.
        spreadBuyQuote = self.state.quoteState[quoteKey]
        legsKeys = list(filter(lambda x: not isinstance(x, tuple), qk))

        legsquotes = [self.state.quoteState[legKey] for legKey in legsKeys]
        for i in range(0, 15):
            # when not populated, bid/ask are nan
            # bid = speadBuyQuote.bid
            # ask = speadBuyQuote.ask

            # construct our own spread quote because the IBKR spread datafeed is unreliable.
            # This is just a fully long spread, so the quotes are just the sum of each side.
            bid = 0
            ask = 0
            for leg in legsquotes:
                bid += leg.bid
                ask += leg.ask

            middle = (bid + ask) / 2

            # but bid/ask can also be 0  or -1 when no trade is viable (too wide 0DTE spreads with no buyers, etc)
            # (this is okay because these are LONG straddles/strangles so we are not running credit orders at all)
            isok = middle and middle == middle and middle > 0

            if isok:
                middle = round(comply(self.tradeSymbol, middle), 2)
                logger.info(
                    "[{} :: {}] Current quote: [bid {:.2f}] [mid {:.2f}] [ask {:.2f}]",
                    self.symbol,
                    self.tradeSymbol,
                    bid,
                    middle,
                    ask,
                )
                if not (bid == bid or bid):
                    logger.warning(
                        "[{} :: {}] Spread does not have bids for all legs!",
                        self.symbol,
                        self.tradeSymbol,
                    )
                break

            logger.warning("[{} :: {}] No quote yet...", self.symbol, self.tradeSymbol)

            # else, price not found yet, so wait to try again.
            await asyncio.sleep(0.222)

        if not isok:
            logger.error(
                "[{} :: {}] Spread doesn't have a quote? Can't continue!",
                self.symbol,
                self.tradeSymbol,
            )
            return None

        # TODO: for now we're being lazy and assuming a 100 multiplier on everything...
        usePrice = middle  # could also be speadBuyQuote.bid or speadBuyQuote.ask or other ranges

        if not usePrice:
            logger.warning(
                "[{} :: {}] Price not found! Can't continue.",
                self.symbol,
                self.tradeSymbol,
            )
            return

        # make sure we don't try to submit a negative quantity, because we are not GOING SHORT here.
        qty = int(self.amount // (usePrice * 100))

        if qty <= 0:
            logger.warning(
                "Why did your quantity not work? Created quantity {} from price {}",
                qty,
                usePrice,
            )
            return None

        logger.info(
            "[{} :: {}] Decided on offer: {:.2f} with qty {:,} (est: ${:,.2f} before commissions)",
            self.symbol,
            self.tradeSymbol,
            usePrice,
            qty,
            qty * usePrice * 100,
        )

        algo = "LIM"
        useAlgoFromMapLookup = ALGOMAP[algo]

        # even if the spread doesn't have a quote, we still need to use the Bag contract with the legs inside
        bag = spreadBuyQuote.contract

        if False:
            order = orders.IOrder("BUY", qty, usePrice, outsiderth=True).order(
                useAlgoFromMapLookup
            )

            assert isinstance(
                bag, Bag
            ), f"Why isn't your bag a bag? Your contract is: {bag}"

            logger.info(
                "[{} :: {}] BAG: {} via ORDER: {}",
                self.symbol,
                self.tradeSymbol,
                pp.pformat(bag),
                pp.pformat(order),
            )

        # Let the common order executor handle the remainder of this actual trade or preview state
        return await self.state.placeOrderForContract(
            self.symbol,
            True,  # spreads are always "BUY" orders
            bag,
            PriceOrQuantity(qty, is_quantity=True),
            limit=usePrice,
            orderType=useAlgoFromMapLookup,
            preview=self.preview,
        )

        # If we were previewing and ordering locally...
        if self.preview:
            trade = await self.ib.whatIfOrderAsync(bag, order)
            logger.info("PREVIEW: {}", pp.pformat(trade))
            logger.warning("ONLY PREVIEW. NO TRADE PLACED.")
            return

        if False:
            if not contract.exchange:
                contract.exchange = "SMART"

            trade = self.ib.placeOrder(contract, order)


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
                self.symbol, contract = self.state.quoteResolve(self.symbol)
                if not self.symbol:
                    logger.error("[{}] Symbol not found for mapping?", initSym)
                    return None

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
        now = pendulum.now().in_tz("US/Eastern")

        # if after market close, use next day
        if (now.hour, now.minute) >= (16, 15):
            now = now.add(days=1)

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

            if any(np.isnan([currentLow, currentPrice])):
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

        if np.isnan([currentPrice]):
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

        buyQty = defaultdict(int)

        remaining = self.amount
        spend = 0
        skip = 0
        lastPrice = 0
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
                        # if ask is populated, skip rest of waiting
                        # Note: these asks only work for longs! if we have
                        # shot quotes with negative asks, all of this breaks.
                        # We check for (ask > 0) because when IBKR has no live
                        # information (or on init) it returns "ask -1" for a while,
                        # which blows up our calculations obviously.
                        if (not np.isnan(ask)) and ask > 0:
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

        return True


@dataclass
class IOpOrderLimit(IOp):
    """Create a new buy or sell order using limit, market, or algo orders."""

    def argmap(self):
        # allow symbol on command line, optionally
        # symbols are retrieved using just self.args instead of being direct
        # required params here.
        return []

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
                            self.state.quoteResolve(x)[0] if x.startswith(":") else x
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

            contract = None
            portItem = None  # portfolioItem
            # if sym is a CONTRACT, make it explicit and also
            # retain the symbol name independently
            if isinstance(symInput, str):
                sym = symInput
            else:
                portItem = symInput
                contract: Contract = portItem.contract
                sym: str = contract.symbol

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
            await self.state.qualify(contract)

        return await self.state.placeOrderForContract(
            sym,
            isLong,
            contract,
            PriceOrQuantity(qty, is_quantity=True),
            price,
            orderType,
        )

        # TODO: allow opt-in to midpoint price adjustment following.


@dataclass
class IOpCachedQuote(IOp):
    """Return last cached value of a subscribed quote symbol."""

    def argmap(self):
        return [DArg("*symbols")]

    async def run(self):
        for sym in self.symbols:
            # if asked in "future format", drop the slash
            # TODO: should the slash removal be inside currentQuote?
            # TODO: should the slashes be part of the quote symbol name anyway?
            if sym.startswith("/"):
                sym = sym[1:]

            self.state.currentQuote(sym)


@dataclass
class IOpOrderCancel(IOp):
    """Cancel waiting orders via order ids or interactive prompt."""

    def argmap(self):
        return [
            DArg(
                "*orderids",
                lambda xs: [int(x) if x.isdigit() else x for x in xs],
                errmsg="Order IDs can be order id integers or glob strings for symbols to cancel.",
            )
        ]

    async def run(self):
        # If order ids not provided via command, show a selectable list of all orders
        if not self.orderids:
            # We need to manually declare the bag multipler of 100, which is assuming the contract is using standard 100 multipler spreads
            # and not weird things like 40 multipler spreads. ymmv, but it's the best we can do for now without using a lookup table
            # mapping for matching underlyings to multiplier mappings.
            ords = [
                CB(
                    "Orders to Cancel",
                    choices=[
                        Choice(
                            f"{o.order.action} {o.contract.localSymbol} ({o.order.totalQuantity} * ${o.order.lmtPrice:.2f}) == ${o.order.totalQuantity * o.order.lmtPrice * float(o.contract.multiplier or (100 if isinstance(o.contract, Bag) else 1)):,.6f}",
                            o.order,
                        )
                        for o in sorted(
                            filter(
                                lambda x: x.orderStatus.clientId == self.state.clientId,
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
            for orderid in self.orderids:
                # These aren't indexed in anyway, so just n^2 search, but the
                # count is likely to be tiny overall.
                # TODO: we could maintain a cache of active orders indexed by orderId
                for o in self.ib.openTrades():
                    # we can't cancel orders not on our current orderId (clientId==0 can see all orders, but it can't modify them)
                    if o.orderStatus.clientId != self.state.clientId:
                        continue

                    # if provided direct order id integer, just check directly...
                    if isinstance(orderid, int):
                        if o.order.orderId == orderid:
                            oooo.append(o.order)
                    else:
                        # else, is either a symbol name or glob to try...
                        name = o.contract.localSymbol.replace(" ", "")

                        # also allow evict :N if we just have a quote for it handy...
                        # TODO: actually move this out to the arg pre-processing step so we don't run it each time
                        if orderid[0] == ":":
                            orderid, _contract = self.state.quoteResolve(orderid)

                        if fnmatch.fnmatch(name, orderid):
                            oooo.append(o.order)

        if not oooo:
            logger.error("[{}] No match for orders cancel!", self.orderids)
            return

        for n, o in enumerate(oooo, start=1):
            logger.info("[{} of {}] Cancel for order: {}", n, len(oooo), o)
            self.ib.cancelOrder(o)


@dataclass
class IOpBalance(IOp):
    """Return the currently cached account balance summary."""

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

    def argmap(self):
        return [DArg("*symbols", convert=set)]

    def totalFrame(self, df, costPrice=False):
        if df.empty:
            return None

        # Add new Total index row as column sum (axis=0 is column sum; axis=1 is row sum)
        totalCols = [
            "position",
            "marketValue",
            "totalCost",
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
            "closeOrderValue",
            "strike",
        ]
        simpleCols = [
            "%",
            "w%",
            "unrealizedPNL",
            "dailyPNL",
            "totalCost",
            "marketValue",
        ]

        # convert columns to all strings because we format them as nice to read money strings, but pandas
        # doesn't like replacing strings over numeric-typed columns anymore.
        df[simpleCols] = df[simpleCols].astype(str)
        df[detailCols] = df[detailCols].astype(str)

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
        for o in ords:  # , key=lambda p: p.contract.localSymbol):
            backQuickRef.append((o.contract.secType, o.contract.symbol, o.contract))

            # Nice debug output showing full contracts.
            # TODO: enable global debug flags for showing these
            # maybe just enable logger.debug mode via a command?
            # logger.info("{}", o.contract)

            make = {}

            # 't' used for switching on OPT/WAR/STK/FUT types later too.
            t = o.contract.secType

            make["type"] = t
            make["sym"] = o.contract.symbol

            # TODO: update this to allow glob matching wtih fnmatch.filter(sourceCollection, targetGlob)
            if self.symbols and make["sym"] not in self.symbols:
                continue

            # logger.info("contract is: {}", o.contract)
            if isinstance(o.contract, Warrant) or isinstance(o.contract, Option):
                try:
                    make["date"] = pendulum.parse(
                        o.contract.lastTradeDateOrContractMonth
                    ).date()
                except:
                    logger.error("Row didn't have a good date? {}", o)
                    pass
                make["strike"] = o.contract.strike
                make["PC"] = o.contract.right
            make["exch"] = o.contract.primaryExchange[:3]
            make["position"] = o.position
            make["marketPrice"] = o.marketPrice

            close = self.state.orderPriceForContract(o.contract, o.position)

            # if it's a list of tuples, break them by newlines for display
            multiplier = float(o.contract.multiplier or 1)
            if isinstance(close, list):
                closingSide = " ".join([str(x) for x in close])
                make["closeOrderValue"] = " ".join(
                    [str(size * price * multiplier) for size, price in close]
                )
            else:
                closingSide = close
                make["closeOrderValue"] = close * o.position * multiplier

            make["closeOrder"] = closingSide
            make["marketValue"] = o.marketValue
            make["totalCost"] = o.averageCost * o.position
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

            if t == "FUT":
                mult = float(o.contract.multiplier or 1)
                make["averageCost"] = o.averageCost / mult
                make["%"] = (o.marketPrice * mult - o.averageCost) / o.averageCost * 100
            elif t == "BAG":
                logger.info("available: {}", o)
            elif t == "OPT":
                # For some reason, IBKR reports marketPrice
                # as the contract price, but averageCost as
                # the total cost per contract. shrug.
                make["%"] = (o.marketPrice * 100 - o.averageCost) / o.averageCost * 100

                # show average cost per share instead of per contract
                # because the "marketPrice" live quote is the quote
                # per share, not per contract.
                make["averageCost"] = o.averageCost / 100
            else:
                make["%"] = (o.marketPrice - o.averageCost) / o.averageCost * 100
                make["averageCost"] = o.averageCost

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
                "PC",
                "date",
                "strike",
                "exch",
                "position",
                "averageCost",
                "marketPrice",
                "closeOrder",
                "closeOrderValue",
                "marketValue",
                "totalCost",
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
        symbolCounts = df.pivot_table(index=["type", "sym"], aggfunc="size")

        spreadSyms = set()
        for (postype, sym), symCount in symbolCounts.items():
            if postype == "OPT" and symCount > 1:
                spreadSyms.add(sym)

        # print individual frames for each spread since the summations
        # will look better
        for sym in spreadSyms:
            spread = df[(df.type == "OPT") & (df.sym == sym)]
            spread = self.totalFrame(spread.copy(), costPrice=True)
            printFrame(spread, f"[{sym}] Potential Spread Identified")

            matchingContracts = [
                contract
                for type, bqrsym, contract in backQuickRef
                if type == "OPT" and bqrsym == sym
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

        # Fields: https://interactivebrokers.github.io/tws-api/classIBApi_1_1Order.html
        # Note: no sort here because we sort the dataframe before showing
        for o in ords:
            # don't print canceled/rejected/inactive orders
            if o.log[-1].status == "Inactive":
                continue

            make = {}
            log = {}

            # if symbol filtering requested, only show requested symbols
            if self.symbols and o.contract.symbol not in self.symbols:
                continue

            def populateSymbolDetails(target):
                target["id"] = o.order.orderId
                target["sym"] = o.contract.symbol
                parseContractOptionFields(o.contract, target)
                target["occ"] = (
                    o.contract.localSymbol.replace(" ", "")
                    if len(o.contract.localSymbol) > 15
                    else ""
                )

            populateSymbolDetails(make)
            populateSymbolDetails(log)

            make["xchange"] = o.contract.exchange
            make["action"] = o.order.action
            make["orderType"] = o.order.orderType
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
            totalMultiplier = 0
            if o.contract.secType == "BAG":
                # is spread, so we need to print more details than just one strike...
                myLegs: list[str] = []

                for x in o.contract.comboLegs:
                    xcontract = self.state.conIdCache.get(x.conId)

                    # if ID -> Name not in the cache, create it
                    if not xcontract:
                        xcontract = await self.state.qualify(Contract(conId=x.conId))

                    totalMultiplier += float(xcontract.multiplier or 1)

                    # now the name will be in the cache!
                    lsym = self.state.conIdCache[x.conId].localSymbol
                    lsymsym, *lsymrest = lsym.split()

                    # leg of spread is an IBKR OPTION SYMBOL (SPY 20240216C00500000)
                    if lsymrest:
                        lsymrest = lsymrest[0]
                        myLegs.append(
                            (
                                x.action[0],  # 0
                                x.ratio,  # 1
                                # Don't need symbol because row has symbol...
                                # lsymsym,
                                lsymrest[-9],  # 2
                                str(pendulum.parse("20" + lsymrest[:6]).date()),  # 3
                                round(int(lsymrest[-8:]) / 1000, 2),  # 4
                                lsym.replace(" ", ""),  # 5
                            )
                        )
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

                make["legs"] = myLegs

            # extract common fields for re-use below
            multiplier = totalMultiplier or float(o.contract.multiplier or 1)
            lmtPrice = float(o.order.lmtPrice or 0)
            totalQuantity = float(o.order.totalQuantity)
            pq = lmtPrice * totalQuantity * multiplier
            make["lreturn"] = 0
            make["lcost"] = 0

            # record whether this order value is a credit into or debit from the account
            if o.order.action == "SELL":
                # IBKR 'sell' prices are always positive and represents a credit back to the account when executed
                assert (
                    lmtPrice > 0
                ), f"How is your order selling price negative? Order: {o.order}"

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
                    "xchange", "orderType",
                    "qty", "cashQty", "filled", "rem", "lmt", "aux", "trail", "tif",
                    "4-8", "lreturn", "lcost", "occ", "legs"],
                )

        # fmt: on
        if df.empty:
            logger.info(
                "{}No open orders exist for client id {}!",
                f"[{", ".join(sorted(self.symbols))}] " if self.symbols else "",
                ICLI_CLIENT_ID,
            )
            return

        df.sort_values(
            by=["date", "sym", "action", "PC", "strike"],
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
                    pendulum.instance(l.time).in_tz("US/Eastern"),
                    l.status,
                    l.message,
                )

        # now print the actual open orders
        printFrame(df)


@dataclass
class IOpExecutions(IOp):
    """Display all executions including commissions and PnL."""

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
            df = pd.DataFrame(l)
            if df.empty:
                logger.info("No {}", name)
            else:
                # if symbol filter requested, remove non-matching contracts.
                # NOTE: we filter on SYMBOL and not "localSymbol" so we don't currently match on extended details like OCC symbols.
                if self.symbols and name == "Contracts":
                    df = df[df.symbol.isin(self.symbols)]

                use.append((name, df))

        if not use:
            return None

        df = pd.concat({name: frame for name, frame in use}, axis=1)

        # Goodbye multiindex...
        df.columns = df.columns.droplevel(0)

        # clean up any non-matching values due to symbols filtering
        if self.symbols:
            df = df[df.conId.notna()]

        # Remove duplicate columns...
        df = df.loc[:, ~df.columns.duplicated()]

        # convert to Eastern time and drop date (since these reports only show up for one day, it's all duplicate details)
        df["time"] = df["time"].apply(pd.Timestamp).dt.tz_convert("US/Eastern")
        df["timestamp"] = df["time"]

        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        df["time"] = df["time"].dt.strftime("%H:%M:%S")

        df["c_each"] = df.commission / df.shares

        # really have to stuff this multiplier change in there due to pandas typing requirements
        df.multiplier = df.multiplier.replace("", 1).fillna(1).astype(float)

        df["total"] = round(df.shares * df.avgPrice, 2) * df.multiplier

        # Note: 'realizedPNL' for the closing transactions *already* includes commissions for both the buy and sell executions,
        #       so *don't* subtract commissions again anywhere.
        df["dayProfit"] = df.realizedPNL.cumsum()

        df["RPNL_each"] = df.realizedPNL / df.shares

        # provide a weak calculation of commission as percentage of PNL and of traded value.
        # this isn't entirely accurate because it doesn't account for the _entry_ commission, we are only
        # using the exit commission when a realized PNL value is provided.
        df["c_pct"] = df.commission / (df.total + df.commission)
        df["c_pct_RPNL"] = df.commission / (df.realizedPNL + df.commission)

        dfByTrade = df.groupby("orderId side localSymbol".split()).agg(
            dict(
                date=["min"],
                time=[("start", "min"), ("finish", "max")],
                price=["mean"],
                shares=["sum"],
                total=["sum"],
                commission=["sum"],
                # TODO: we need to calculate a manlal c_each by (total commision / shares) instead of mean of the c_each directly
                c_each=["mean"],
            )
        )

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

        df.rename(columns={"lastTradeDateOrContractMonth": "date"}, inplace=True)
        # ignoring: "execId" (long string for execution recall) and "permId" (???)

        # removed: lastLiquidity avgPrice
        df = df[
            (
                """secType conId strike right date exchange symbol tradingClass localSymbol time orderId
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

        profitByHour = dfByTimeProfit.resample("30Min").agg(
            dict(realizedPNL="sum", orderId="count", profit="count", loss="count")
        )

        # TODO: format realizedPNL and dayProfit as prices for profitByHour
        profitByHour.rename(columns=dict(orderId="executions"), inplace=True)
        profitByHour["dayProfit"] = profitByHour.realizedPNL.cumsum()

        needsPrices = "realizedPNL dayProfit".split()
        profitByHour[needsPrices] = profitByHour[needsPrices].map(fmtPrice)

        desc = ""
        if self.symbols:
            desc = f" Filtered for: {', '.join(self.symbols)}"

        printFrame(df, f"Execution Summary{desc}")
        printFrame(profitByHour, f"Profit by Half Hour{desc}")
        printFrame(
            dfByTrade.sort_values(
                by=[("date", "min"), ("time", "start"), "orderId", "localSymbol"]
            ),
            f"Execution Summary by Complete Order{desc}",
        )


@dataclass
class IOpQuotesAdd(IOp):
    """Add live quotes to display."""

    def argmap(self):
        return [DArg("*symbols", convert=lambda x: expand_symbols(x))]

    async def run(self):
        return await self.state.addQuotes(self.symbols)


@dataclass
class IOpQuotesAddFromOrderId(IOp):
    """Add symbols for current orders to the quotes view."""

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
            tickFields = tickFieldsForContract(useTrade.contract)

            # If this is a new session and we haven't previously cached the
            # contract id -> name mappings, we need to look them all up now
            # or else the next print of the quote toolbar will throw lots
            # of missing key exceptions when trying to find names.
            if useTrade.contract.comboLegs:
                for x in useTrade.contract.comboLegs:
                    # if ID -> Name not in the cache, create it
                    if not x.conId in self.state.conIdCache:
                        await self.state.qualify(Contract(conId=x.conId))
            else:
                if useTrade.contract.conId not in self.state.conIdCache:
                    await self.state.qualify(Contract(conId=useTrade.contract.conId))

            self.state.addQuoteFromContract(useTrade.contract)


@dataclass
class IOpQuotesRemove(IOp):
    """Remove live quotes from display."""

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
        for sym in sorted(set(self.symbols), reverse=True):
            sym = sym.upper()
            if len(sym) > 30:
                # this is a combo request, so we need to evaluate, resolve, then key it
                orderReq = self.state.ol.parse(sym)
                contract = await self.state.contractForOrderRequest(orderReq)
            elif sym[0] == ":":
                resolved, contract = self.state.quoteResolve(sym)
                if not resolved:
                    logger.warning(
                        "[{}] No matching symbol index found? No quote to remove!",
                        sym,
                    )
                    continue

                # symbol is now the replaced actual symbol and not the integer-indexed reference
                sym = resolved
            else:
                found = self.state.quoteState.get(sym)
                if found:
                    contract = found.contract
                else:
                    logger.warning("[{}] Symbol not found as an active quote?", sym)
                    continue

            if contract:
                try:
                    self.ib.cancelMktData(contract)

                    symkey = lookupKey(contract)
                    del self.state.quoteState[symkey]
                    del self.state.ema[symkey]
                    logger.info(
                        "[{}] Removed: {} ({})",
                        sym,
                        contract.localSymbol or contract.symbol,
                        symkey,
                    )
                except:
                    # user requested removal of non-subscribed quote
                    # (which is still okay)
                    # logger.exception("no go?")
                    pass


@dataclass
class IOpOrderSpread(IOp):
    """Place a spread order described by using BuyLang/OrderLang"""

    def argmap(self):
        return [DArg("legdesc"), DArg("*preview")]

    async def run(self):
        bag = None

        logger.info("NOTE: SELLING FOR CREDIT requires NEGATIVE PRICE")

        # if we're using a quote as entrypoint, look it up first...
        if self.legdesc.startswith(":"):
            desc = self.legdesc
            # this logic is manually here instead of reusing the cli.py abstraction because
            # the contract here is a bag which has different properties...
            lookupInt = int(self.legdesc[1:])
            try:
                name, ticker = self.state.quotesPositional[lookupInt]
                bag = ticker.contract
            except:
                logger.error("No matching quote index for {}", self.legdesc)
                return

        promptPosition = [
            # Only ask for the symbol if we're not providing a symbol ourself
            Q("Symbol", value=self.legdesc) if not bag else None,
            Q("Price"),
            Q("Quantity"),
            ORDER_TYPE_Q,
        ]

        if not bag and self.legdesc:
            await self.runoplive(
                "add",
                f'"{self.legdesc}"',
            )

        got = await self.state.qask(promptPosition)

        try:
            if not bag:
                req = got["Symbol"]
                desc = req
                orderReq = self.state.ol.parse(req)

                # re-add quote if changed or new (no impact if already added)
                asyncio.create_task(self.runoplive("add", f'"{req}"'))

            qty = int(got["Quantity"])
            price = float(got["Price"])
            orderType = got["Order Type"]
            # It appears spreads with IBKR always have "BUY" order action, then the
            # credit/debit is addressed by negative or positive prices.
            # (i.e. you can't "short" a spread and I guess closing the spread is
            #       just an "inverse BUY" in their API's view)
            # also TIF limits due to algos only operating RTH... (TODO: this also needs to just check for SPX/VIX options for outside=True)
            order = orders.IOrder(
                "BUY",
                qty,
                price,
                outsiderth=True if orderType in {"LMT", "MKT", "MIT"} else False,
            ).order(orderType)
        except Exception as e:
            logger.warning("Order canceled due to incomplete fields: {}", e)
            return None

        # only calculate a bag from the order request if this isn't :N index request symbol
        if not bag:
            bag = await self.state.bagForSpread(orderReq)

        if self.preview:
            logger.info(
                "[{}] PREVIEW REQUEST {} via {}",
                desc,
                pp.pformat(bag),
                pp.pformat(order),
            )

            trade = await asyncio.wait_for(
                self.ib.whatIfOrderAsync(bag, order), timeout=2
            )

            logger.info("[{}] TRADE PREVIEW: {}", desc, pp.pformat(trade))
            logger.warning("[{}] ONLY PREVIEW. NO TRADE PLACED.", desc)
            # TODO: add more preview calculations around commissions, etc like we do with the regular single order previews.
            # TOTO: also maybe move this entire execution logic into cli.py next to the regular single-contract order processing?
            return

        trade = self.ib.placeOrder(bag, order)
        logger.info("TRADE EXECUTED: {}", pp.pformat(trade))


@dataclass
class IOpOptionChainRange(IOp):
    """Print OCC symbols for an underlying with ± offset for puts and calls."""

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
        currentPrice = fetchone[np.isfinite(fetchone)][0]

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
        now = pendulum.now("US/Eastern")
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

    def argmap(self):
        # TODO: add options to allow multiple symbols and also printing cached chains
        return [
            DArg(
                "*symbols",
                convert=lambda x: [z.upper() for z in x],
            )
        ]

    async def run(self):
        # TODO: have this fech from optional external API because IBKR chains API is
        # excessively rate limited garbage.

        # Index cache by symbol AND current date because strikes can change
        # every day even for the same expiration if there's high volatility.
        now = pendulum.now("US/Eastern")
        got = {}

        IS_PREVIEW = self.symbols[-1] == "PREVIEW"
        for symbol in self.symbols:
            # if asking for weeklies, need to correct symbol for underlying quote...
            if symbol == "SPXW":
                symbol = "SPX"

            # our own symbol hack for printing more detailed output here
            if symbol == "PREVIEW":
                continue

            if symbol.startswith(":"):
                symbol, _ = self.state.quoteResolve(symbol)
                assert symbol

            cacheKey = ("chains", symbol, now.date())
            # logger.info("Looking up {}", cacheKey)

            # if found in cache, don't lookup again!
            if True:
                if found := self.cache.get(cacheKey):
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

            # note "fetching" here so it doesn't fire if the cache hit passes.
            # (We only want to log the fetching network requests; just cache reading is quiet)
            logger.info("[chains] Fetching for {}", symbol)

            # resolve for option chains lookup
            contractExact = contractForName(symbol)

            # if we want to request ALL chains, only provide underlying
            # symbol then mock it as "OPT" so the IBKR resolver sees we
            # are requesting option underlying and not just single-stock
            # contract details.
            contractExact.secType = "OPT"

            # If full option symbol, get all strikes for the date of the symbol
            if isinstance(contractExact, (Option, FuturesOption)):
                contractExact.strike = 0.00
                # if is option already, use exact date of option...
                useDates = [pendulum.parse("20" + symbol[-15 : -15 + 6]).date()]
            else:
                # We try to collect 3 days for our chains fetching:
                #   - THIS month
                #   - month of the Friday of this week
                #   - If we are past the 15th of the month, also fetch NEXT month for next opex calculations.
                #   - but, since this is a set, these will be reduced to either 1 or 2 total values
                #     - the "check friday" technically isn't needed with the +20 day check, but it's logically
                #       what is needed for the data we are trying to fetch too.
                # Also this is a weird hack because IBKR has awful data feeds and these chain fetches take
                # multiple minutes due to IBKR "pacing" rate limiting, while fetching chains from traider or
                # polygon takes milliseconds for hundreds of chains.
                # TODO: we should allow alternate chain lookup feeds or using a local cache populated from
                #       external services.
                useDates = set(
                    [
                        pendulum.date(d.year, d.month, 1)
                        for d in [
                            now,
                            pendulum.parse("friday", strict=False),
                            (now + datetime.timedelta(days=20)),
                        ]
                    ]
                )

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
            strikes: dict[str, list[float]] = defaultdict(list)
            got[symbol] = strikes

            for d in sorted(useDates):
                if symbol.startswith("/"):
                    # Futures use future expiration
                    contractExact.lastTradeDateOrContractMonth = FUT_EXP
                else:
                    contractExact.lastTradeDateOrContractMonth = f"{d.year}{d.month:02}"

                logger.info(
                    "[{}{}] Fetching strikes...",
                    symbol,
                    contractExact.lastTradeDateOrContractMonth,
                )

                # TODO: replace with reqSecDefOptParams()?
                chainsExact = await self.ib.reqContractDetailsAsync(contractExact)

                # logger.info("Full result: {}", chainsExact)

                # group strike results by date
                logger.info(
                    "[{}{}] Populating strikes...",
                    symbol,
                    contractExact.lastTradeDateOrContractMonth,
                )

                for d in sorted(
                    chainsExact, key=lambda k: k.contract.lastTradeDateOrContractMonth
                ):
                    strikes[d.contract.lastTradeDateOrContractMonth].append(
                        d.contract.strike
                    )

            # cleanup the results because they were received in an
            # arbitrary order, but we want them sorted for bisecting
            # and just nice viewing.
            for k, v in strikes.items():
                assert k
                assert v
                # also reduce to a set first to drop all the duplicate
                # call/put strikes.
                strikes[k] = sorted(set(v))

            # logger.info("Saving into {}", cacheKey)

            # expire strike caches at the next 1615 available
            # TODO: remove date() from cache key and only rely on .expire() instead?
            # compare now.time() against pendulum.Time(16, 15, 0)
            self.cache.set(cacheKey, strikes, expire=86400)

            logger.info("Strikes: {}", pp.pformat(strikes))

            if False:
                df = pd.DataFrame(chainsExact)
                printFrame(df)

        return got


@dataclass
class IOpQuoteSave(IOp):
    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        self.cache.set(cacheKey, set(self.symbols))
        logger.info("[{}] {}", self.group, self.symbols)

        repopulate = [f'"{x}"' for x in self.symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )


@dataclass
class IOpQuoteAppend(IOp):
    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            logger.error(
                "[{}] No quote group found. Creating new quote group!", self.group
            )
            symbols = set()

        self.cache.set(cacheKey, symbols | set(self.symbols))
        repopulate = [f'"{x}"' for x in self.symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )


@dataclass
class IOpQuoteRemove(IOp):
    """Remove symbols from a quote group."""

    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        nocache = False
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            nocache = True
            symbols = self.state.quoteState.keys()
            logger.error(
                "[{}] No quote group found so using live quote list...", self.group
            )

        goodbye = set()
        for s in self.symbols:
            for symbol in symbols:
                # guard against running fnmatch on the tuple entries we use for spread quotes
                # (spread quotes can only be removed by :N id)
                if isinstance(symbol, str):
                    if fnmatch.fnmatch(symbol, s):
                        logger.info("Dropping quote: {}", symbol)
                        goodbye.add(symbol)

        symbols -= goodbye

        # don't *CREATE* the cache key if we didn't use the cache anyway
        if not nocache:
            self.cache.set(cacheKey, symbols)

        repopulate = goodbye | {
            self.state.symbolNormalizeIndexWeeklyOptions(f'"{x}"') for x in goodbye
        }

        logger.info("Removing quotes: {}", ", ".join(repopulate))

        await self.runoplive(
            "remove",
            " ".join(repopulate),
        )


@dataclass
class IOpQuoteRestore(IOp):
    def argmap(self):
        return [DArg("group")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return

        repopulate = [f'"{x}"' for x in symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )


@dataclass
class IOpQuoteClean(IOp):
    def argmap(self):
        return [DArg("group")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)
        if not symbols:
            logger.error("No quote group found for name: {}", self.group)
            return

        # Find any expired option symbols and remove them
        remove = []
        now = pendulum.now().in_tz("US/Eastern")

        # if after market close, use today; else use previous day since market is still open
        if (now.hour, now.minute) < (16, 15):
            now = now.subtract(days=1)

        datecompare = now.strftime("%y%m%d")
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
            groups = sorted([k[1] for k in self.cache if k[0] == "quotes"])

        logger.info("Groups: {}", pp.pformat(groups))

        for group in groups:
            logger.info(
                "[{}] Members: {}",
                group,
                pp.pformat(sorted(self.cache.get(("quotes", group), []))),
            )


@dataclass
class IOpQuoteGroupDelete(IOp):
    """Delete an entire quote group"""

    def argmap(self):
        return [DArg("*groups", convert=set, desc="quote group names to delete.")]

    async def run(self):
        if not self.groups:
            logger.error("No groups provided!")

        for group in sorted(self.groups):
            logger.info("Deleting quote group: {}", group)
            self.cache.delete(("quotes", group))


# TODO: potentially split these out into indepdent plugin files?
OP_MAP = {
    "Live Market Quotes": {
        "qquote": IOpQQuote,
        "quote": IOpCachedQuote,
        "depth": IOpDepth,
        "add": IOpQuotesAdd,
        "oadd": IOpQuotesAddFromOrderId,
        "remove": IOpQuotesRemove,
        "rm": IOpQuotesRemove,
        "chains": IOpOptionChain,
        "range": IOpOptionChainRange,
    },
    "Order Management": {
        "limit": IOpOrderLimit,
        "buy": IOpOrder,
        "fast": IOpOrderFast,
        "spread": IOpOrderSpread,
        "modify": IOpOrderModify,
        "evict": IOpPositionEvict,
        "cancel": IOpOrderCancel,
        "straddle": IOpOrderStraddle,
    },
    "Portfolio": {
        "balance": IOpBalance,
        "positions": IOpPositions,
        "ls": IOpPositions,
        "orders": IOpOrders,
        "executions": IOpExecutions,
    },
    "Connection": {
        "rid": IOpRID,
    },
    "Utilities": {
        "rcheck": None,
        "future": None,
        "bars": None,
        "try": None,
        "tryf": None,
        "cash": IOpCash,
        "alias": IOpAlias,
        "alert": IOpAlert,
        "calendar": IOpCalendar,
        "math": IOpCalculator,
        "info": IOpInfo,
        "expand": IOpExpand,
        "set": IOpSetEnvironment,
        "say": IOpSay,
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
    "Quote Management": {
        "qsave": IOpQuoteSave,
        "qadd": IOpQuoteAppend,
        "qremove": IOpQuoteRemove,
        "qrm": IOpQuoteRemove,
        "qdelete": IOpQuoteGroupDelete,
        "qrestore": IOpQuoteRestore,
        "qclean": IOpQuoteClean,
        "qlist": IOpQuoteList,
    },
}


@dataclass
class Dispatch:
    def __post_init__(self):
        self.d = mutil.dispatch.Dispatch(OP_MAP)

    def runop(self, *args, **kwargs) -> Coroutine:
        return self.d.runop(*args, **kwargs)
