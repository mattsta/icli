"""Shared utilities and helper functions for commands.

This module contains common utilities extracted from lang.py that are
used across multiple commands.
"""

# Imports for shared utilities
from typing import TYPE_CHECKING

import mutil.expand
import pandas as pd
from questionary import Choice

from icli.helpers import Q

if TYPE_CHECKING:
    from ib_async import FuturesOption, Option


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
