""" A refactor-base for splitting out common helpers between cli and lang """

import ib_insync  # just for UNSET_DOUBLE
from ib_insync import Stock, Future, Option, Warrant, FuturesOption

from icli.futsexchanges import FUTS_EXCHANGE
import pandas as pd
import numpy as np
import pendulum

from dataclasses import dataclass, field
import questionary
from questionary import Choice

from typing import *

from loguru import logger
import shutil
from dotenv import dotenv_values
import os

# TODO: detect this automatically:
FU_DEFAULT = dict(ICLI_FUT_EXP=202109)
FU_CONFIG = {**FU_DEFAULT, **dotenv_values(".env.icli"), **os.environ}

# FUT_EXP = "202106"
FUT_EXP = FU_CONFIG["ICLI_FUT_EXP"]


def contractForName(sym, exchange="SMART", currency="USD"):
    """Convert a single text symbol into an ib_insync contract.

    Text symbols are assumed to be one of:
        - Future if symbol begins with '/'
        - Option if symbol is > 15 characters long
            - Future Option if symbol is large *and* starts with '/'
        - Stock otherwise
    """
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
        )
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

    Sorting is also flexible where if no date is available, the sort still works fine."""

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
    return (makeQuarter(lower), makeQuarter(upper))


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


def makeQuarter(x) -> float:
    # TODO: replace with mutil.numeric.roundnear for 0.25 and 1

    """Convert any price to end in .00, 0.25, 0.50, or 0.75 to match
    the tick intervals required for futures.

    MES ticks by $0.25, but MYM ticks only by $1, MNQ ticks by $0.25

    Note: this rounds everything down. Could use ceil to round up if necessary."""
    return round(round(x * 4) / 4, 2)


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
                # **kwargs,
            ).ask_async()

        return questionary.text(self.msg, **kwargs).ask_async()
