""" A refactor-base for splitting out common helpers between cli and lang """

from dataclasses import dataclass, field

import ib_insync  # just for UNSET_DOUBLE
import numpy as np
import pandas as pd
import pendulum
import questionary
import tradeapis.cal as tcal
import tradeapis.rounder as rounder
from ib_insync import (
    Bond,
    CFD,
    Contract,
    Crypto,
    Forex,
    Future,
    FuturesOption,
    Index,
    Option,
    Stock,
    Warrant,
)
from questionary import Choice

from icli.futsexchanges import FUTS_EXCHANGE


from typing import *
import datetime
import os
import shutil

from dotenv import dotenv_values

from loguru import logger

# auto-detect next index futures expiration month based on roll date
# we add some padding to the futs exp to compensate for having the client open a couple days before
# (which will be weekends or sunday night, which is fine)
futexp = tcal.nextFuturesRollDate(
    datetime.datetime.now().date() + datetime.timedelta(days=3)
)

# Also compare: https://www.cmegroup.com/trading/equity-index/rolldates.html
logger.info("Futures Next Roll-Forward Date: {}", futexp)
FU_DEFAULT = dict(ICLI_FUT_EXP=f"{futexp.year}{futexp.month:02}")  # YM like: 202309
FU_CONFIG = {**FU_DEFAULT, **dotenv_values(".env.icli"), **os.environ}  # type: ignore

FUT_EXP = FU_CONFIG["ICLI_FUT_EXP"]


def comply(contract: Union[Contract, str], price: float) -> float:
    """Conform a calculated price to an IBKR-acceptable price increment.

    We say "IBKR-acceptable" because for some price increments IBKR will self-adjust
    internally, while for other proeducts it requires exact conformity. shrug."""

    if isinstance(contract, Future):
        # ES/MES/NQ/MNQ futures have a 0.25 minimum tick increment and other futures
        # have different increments, to use an API to figure it out.
        # TODO: is "symbol" the root symbol like "ES" or the time symbol like ESU2?
        logger.info("[/{}] ROUNDING REQUEST: {}", contract.symbol, price)
        rounded = rounder.round("/" + contract.symbol, price)
        logger.info("[/{}] ROUNDED: {}", contract.symbol, rounded)
        return rounded

    if isinstance(contract, (Crypto, Forex)):
        logger.info("[{}] ROUNDING REQUEST: {}", contract.symbol, price)
        rounded = rounder.round(contract.symbol, price)
        logger.info("[{}] ROUNDED: {}", contract.symbol, rounded)
        return rounded

    if isinstance(contract, Option):
        # hack for SPX or other index options needing specific increments
        # (IBKR API for contract details is slow and busted, so we either need to have a
        #  local DB of symbols to increments or just do minimal hacks like these along the way...)

        # note: IBKR handles "penny pilot program" options internally so you don't need to comply with tick
        #       increments, but for index options, they DO enforce tick increments, so we need this internal
        #       lookup table hack. thanks, IBKR!

        # "SPX trades in specific increments of $0.05 when premiums are less than $3 and $0.10 for premiums higher than or equal to $3."
        name = contract.localSymbol[:-15].rstrip()
        return rounder.round(name, price)

    # another hack in case we're just doing quotes or something?
    if isinstance(contract, str):
        # if input is a full option symbol, use only the symbol name
        if len(contract) > 10:
            contract = contract[:-15].rstrip()

        return rounder.round(name, price)

    # else, price doesn't match a condition so we remain unchanged
    return price


def contractForName(sym, exchange="SMART", currency="USD"):
    """Convert a single text symbol data format into an ib_insync contract."""

    sym = sym.upper()

    # TODO: how to specify warrants/equity options/future options/spreads/bonds/tbills/etc?
    if sym.startswith("/"):
        sym = sym[1:]
        if len(sym) > 15:
            # Is Future Option! FOP!
            symbol = sym[:-15]

            body = sym[-15:]
            date = "20" + body[:6]
            right = body[-9]  # 'C'

            if right not in {"C", "P"}:
                raise Exception(f"Invalid option format right: {right} in {sym}")

            price = int(body[-8:])  # 320000 (leading 0s get trimmed automatically)
            strike = price / 1000  # 320.0

            fxchg = FUTS_EXCHANGE[symbol]
            contract = FuturesOption(
                currency=currency,
                symbol=fxchg.symbol,
                exchange=fxchg.exchange,
                multiplier=fxchg.multiplier,
                strike=strike,
                right=right,
                lastTradeDateOrContractMonth=date,
            )
        else:
            # else, is regular future

            # our symbol lookup table is the unqualified contract name like "ES" but
            # when trading, the month and year gets added like "ESZ3", so if we have
            # a symbol ending in a digit here, remove the "expiration/year" designation
            # from the string to lookup the actual name.
            if sym[-1].isdigit():
                sym = sym[:-2]

            fxchg = FUTS_EXCHANGE[sym]
            contract = Future(
                currency=currency,
                symbol=fxchg.symbol,
                exchange=fxchg.exchange,
                multiplier=fxchg.multiplier,
                lastTradeDateOrContractMonth=FUT_EXP,
            )
    elif len(sym) > 15:
        # looks like: COIN210430C00320000
        symbol = sym[:-15]  # COIN
        body = sym[-15:]  # 210430C00320000

        # Note: Not year 2100+ compliant!
        # 20 + YY + MM + DD
        date = "20" + body[:6]

        right = body[-9]  # 'C'

        if right not in {"C", "P"}:
            raise Exception(f"Invalid option format right: {right} in {sym}")

        price = int(body[-8:])  # 320000 (leading 0s get trimmed automatically)
        strike = price / 1000  # 320.0

        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=date,
            strike=strike,
            right=right,
            exchange=exchange,
            currency=currency,
            # also pass in tradingClass so options like SPXW220915P03850000 work
            # directly instead of having IBKR guess if we want SPX or SPXW.
            # for all other symbols the underlying trading class doesn't alter
            # behavior (and we don't allow users to specify extra classes yet
            # like if you want to trade on fragemented options chains after
            # reverse splits, etc).
            tradingClass=symbol,
        )
    else:
        # if symbol has a : we are namespacing by type:
        #   - W: - warrant
        #   - C: - crypto
        #   - F: - forex
        #   - B: - bond
        #   - S: - stock (or no contract namespace)
        #   - I: - an index value (VIX, VIN, SPX, etc)
        # Note: futures and future options are prefixed with /
        #       equity options are the full OCC symbol with no prefix
        namespaceParts = sym.split(":")
        if len(namespaceParts) > 1:
            contractNamespace, symbol = namespaceParts
            if contractNamespace == "W":
                # TODO: needs option-like strike, right, multiplier, contract date spec too
                contract = Warrant(
                    symbol=symbol, exchange=exchange, currency=currency, right="C"
                )
                # Requires all the details like:
                # contract = Warrant(conId=504262528, symbol='BGRY', lastTradeDateOrContractMonth='20261210', strike=11.5, right='C', multiplier='1', primaryExchange='NASDAQ', currency='USD', localSymbol='BGRYW', tradingClass='BGRY')
            elif contractNamespace == "C":
                contract = Crypto(symbol=symbol, exchange="PAXOS", currency=currency)
            elif contractNamespace == "B":
                contract = Bond(symbol, exchange, currency)
            elif contractNamespace == "S":
                contract = Stock(symbol, exchange, currency)
            elif contractNamespace == "I":
                # this appears to work fine without specifying the full Index(symbol, exchange) format
                contract = Index(symbol)
            elif contractNamespace == "F":
                # things like F:GBPUSD F:EURUSD (but not F:JPYUSD or F:RMBUSD shrug)
                # also remember C: is CRYPTO not "CURRENCY," so currencies are F for FOREX
                contract = Forex(symbol)
            elif contractNamespace == "CFD":
                # things like CFD:XAUUSD
                contract = CFD(symbol)
        else:
            # TODO: warrants, bonds, bills, etc
            contract = Stock(sym, exchange, currency)

    return contract


def tickFieldsForContract(contract) -> str:
    extraFields = []
    if isinstance(contract, Stock):
        # 104:
        # "The 30-day historical volatility (currently for stocks)."
        # 106:
        # "The IB 30-day volatility is the at-market volatility estimated
        #  for a maturity thirty calendar days forward of the current trading
        #  day, and is based on option prices from two consecutive expiration
        #  months."
        # 236:
        # "Number of shares available to short"
        # "Shortable: < 1.5, not availabe
        #             > 1.5, if shares can be located
        #             > 2.5, enough shares are available (>= 1k)"
        extraFields += [104, 106, 236]

    # yeah, the API wants a CSV for the tick list. sigh.
    tickFields = ",".join([str(x) for x in extraFields])

    # logger.info("[{}] Sending fields: {}", contract, tickFields)
    return tickFields


def parseContractOptionFields(contract, d):
    # logger.info("contract is: {}", o.contract)
    if isinstance(contract, Warrant) or isinstance(contract, Option):
        try:
            d["date"] = pendulum.parse(contract.lastTradeDateOrContractMonth).date()
        except:
            logger.error("Row didn't have a good date? {}", contract)
            return
        d["strike"] = contract.strike
        d["PC"] = contract.right
    else:
        # populate columns for non-contracts/warrants too so the final
        # column-order generator still works.
        d["PC"] = None
        d["strike"] = None
        d["date"] = None


def sortLocalSymbol(s):
    """Given tuple of (occ date/right/strike, symbol) return
    tuple of (occ date, symbol)"""

    return (s[0][:6], s[1])


def portSort(p):
    """sort portfolioItem 'p' by symbol if stock or by (expiration, symbol) if option"""
    s = tuple(reversed(p.contract.localSymbol.split()))
    if len(s) == 1:
        return s

    # if option, sort by DATE then SYMBOL (i.e. don't sort by (date, type, strike) complex
    return sortLocalSymbol(s)


def tradeOrderCmp(o):
    """Return the sort key for a trade representing a live order.

    The goal is to sort by:
        - BUY / SELL
        - DATE (if has date, expiration, option, warrant, etc)
        - SYMBOL

    Sorting is also flexible where if no date is available, the sort still works fine.
    """

    # Sort all options by expiration first then symbol
    # (no impact on symbols without expirations)
    useSym = o.contract.symbol
    useName = useSym
    useKey = o.contract.localSymbol.split()
    useDate = -1

    if useKey:
        useName = useKey[0]
        if len(useKey) == 2:
            useDate = useKey[1]
        else:
            # the 'localSymbol' date is 2 digit years while the 'lastTradeDateOrContractMonth' is
            # four digit years, so to compare, strip the leading '20' from LTDOCM
            useDate = o.contract.lastTradeDateOrContractMonth[2:]

    return (useDate, useSym, useName)


def boundsByPercentDifference(mid: float, percent: float) -> tuple[float, float]:
    """Returns the lower and upper percentage differences from 'mid'.

    Percentage is given as a full decimal percentage.
    Example: 0.25% must be provided as 0.0025"""
    # Obviously doesn't work for mid == 0 or percent == 2, but we shouldn't
    # encounter those values under normal usage.

    # This is just the percentage difference between two prices equation
    # re-solved for a and b from: (a - b) / ((a + b) / 2) = percent difference
    lower = -(mid * (percent - 2)) / (percent + 2)
    upper = -(mid * (percent + 2)) / (percent - 2)
    return (lower, upper)


def strFromPositionRow(o):
    """Return string describing an order (for quick review when canceling orders).

    As always, 'averageCost' field is for some reason the cost-per-contract field while
    'marketPrice' is the cost per share, so we manually convert it into the expected
    cost-per-share average cost for display."""

    useAvgCost = (
        o.averageCost / 100 if len(o.contract.localSymbol) > 15 else o.averageCost
    )
    return f"{o.contract.localSymbol} :: {o.contract.secType} {o.position:,.2f} MKT:{o.marketPrice:,.2f} CB:{useAvgCost:,.2f} :: {o.contract.conId}"


def isset(x: float) -> bool:
    """Sadly, ib_insync API uses FLOAT_MAX to mean "number is unset" instead of
    letting numeric fields be Optional[float] where we could just check for None.

    So we have to directly compare against another value to see if a returned float
    is a _set_ value or just a placeholder for the default value. le sigh."""
    return x != ib_insync.util.UNSET_DOUBLE


@dataclass
class Q:
    """Self-asking series of prompts."""

    name: Optional[str] = None
    msg: Optional[str] = None
    choices: Optional[Sequence] = None
    value: str = field(default_factory=str)

    def __post_init__(self):
        # Allow flexiblity with assigning msg/name if they are just the same
        if not self.msg:
            self.msg = self.name

        if not self.name:
            self.name = self.msg

    def ask(self, **kwargs):
        """Prompt user based on types provided."""
        if self.choices:
            # Note: no kwargs on .select() because .select()
            #       is injecting its own bottom_toolbar for error reporting,
            #       even though it never seems to use it?
            #       See: questionary/prompts/common.py create_inquier_layout()
            return questionary.select(
                message=self.msg,
                choices=self.choices,
                use_indicator=True,
                use_shortcuts=True,
                use_arrow_keys=True,
                use_jk_keys=False,
                # **kwargs,
            ).ask_async()

        return questionary.text(self.msg, default=self.value, **kwargs).ask_async()


@dataclass
class CB:
    """Self-asking series of prompts."""

    name: Optional[str] = None
    msg: Optional[str] = None
    choices: Optional[Sequence] = None

    def __post_init__(self):
        # Allow flexiblity with assigning msg/name if they are just the same
        if not self.msg:
            self.msg = self.name

        if not self.name:
            self.name = self.msg

    def ask(self, **kwargs):
        """Prompt user based on types provided."""
        if self.choices:
            # Note: no kwargs on .select() because .select()
            #       is injecting its own bottom_toolbar for error reporting,
            #       even though it never seems to use it?
            #       See: questionary/prompts/common.py create_inquier_layout()
            return questionary.checkbox(
                message=self.msg,
                choices=self.choices,
                use_jk_keys=False,
                # **kwargs,
            ).ask_async()

        return questionary.text(self.msg, **kwargs).ask_async()
