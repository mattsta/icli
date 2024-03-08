"""A simple(ish) built-in calculator."""

import decimal
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from lark import Lark, Transformer

from loguru import logger

decimal.getcontext().prec = 10  # set precision for Decimal

grammar = """
    start: expr
    ?expr: operation
         | SIGNED_NUMBER -> number
         | ":" INT -> positionlookup
         | ":" CNAME -> portfoliovaluelookup
         | CNAME -> stringlookup
    operation: "(" FUNC expr+ ")"
    FUNC: "+" | "-" | "*" | "/" | "gains"i | "grow"i | "o"i | "r"i

# Use custom 'DIGIT' so we can have underscores as place holders in our numbers
# (this is why we are rebuilding the entire number/float/int hierarchy here too)
DIGIT: "0".."9" | "_"
HEXDIGIT: "a".."f"|"A".."F"|DIGIT

INT: DIGIT+
SIGNED_INT: ["+"|"-"] INT
DECIMAL: INT "." INT? | "." INT

_EXP: ("e"|"E") SIGNED_INT
FLOAT: INT _EXP | DECIMAL _EXP?
SIGNED_FLOAT: ["+"|"-"] FLOAT

NUMBER: FLOAT | INT
SIGNED_NUMBER: ["+"|"-"] NUMBER

%import common.CNAME
%import common.WS
%ignore WS
"""


@dataclass
class CalculatorTransformer(Transformer):
    state: Any

    symbol_to_func = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "gains": "gains",
        "o": "optgains",
        "r": "round",
        "grow": "grow",
    }

    def round(self, value):
        """Convert current value to a rounded integer"""
        target, *decimals = value

        # need to convert the decimal count to an 'int' because the parser generated it
        # as a Decimal() by default using the number() rule below.
        count = int(decimals[0]) if decimals else 0

        return round(value[0], count)

    def positionlookup(self, value):
        """Look up a quote's value using positional :N syntax."""
        key = int(value[0])
        ticker = self.state.quotesPositional[key]

        # ticker[0] is just the symbol name while ticker[1] is the Ticker() object...
        q = ticker[1]

        # during "regular times" there's a bid/ask spread
        if q.bidSize and q.askSize:
            mark = (q.bid + q.ask) / 2
        else:
            # else, over weekends and shutdown times, there's either the last price or the close price
            mark = q.last if q.last == q.last else q.close

        logger.info(
            "[:{} -> {}] Using value: {:,.6f}", key, q.contract.localSymbol, mark
        )
        return Decimal(mark)

    def portfoliovaluelookup(self, value):
        """Look up metadata variable from portfolio for calculation."""
        part = value[0].upper()
        match part:
            case "AF":
                value = self.state.accountStatus["AvailableFunds"]
            case "BP":
                value = self.state.accountStatus["BuyingPower4"]
            case "BP4":
                value = self.state.accountStatus["BuyingPower4"]
            case "BP3":
                value = self.state.accountStatus["BuyingPower3"]
            case "BP2":
                value = self.state.accountStatus["BuyingPower2"]
            case "NL":
                value = self.state.accountStatus["NetLiquidation"]
            case "UPL":
                value = self.state.accountStatus["UnrealizedPnL"]
            case "RPL":
                value = self.state.accountStatus["RealizedPnL"]
            case "GPV":
                value = self.state.accountStatus["GrossPositionValue"]
            case "MMR":
                value = self.state.accountStatus["MaintMarginReq"]
            case "EL":
                value = self.state.accountStatus["ExcessLiquidity"]
            case "SMA":
                value = self.state.accountStatus["SMA"]
            case "ELV" | "EWL" | "EWLV":
                # allow as ELV or EWLV
                value = self.state.accountStatus["EquityWithLoanValue"]
            case _:
                logger.error(
                    "[{}] Invalid account variable requested! Calculation can't continue!",
                    part,
                )
                return None

        logger.info("[{}] Using value: {:,.6f}", part, value)
        return Decimal(value)

    def stringlookup(self, value):
        """Attempt to lookup by symbol name directly."""
        part = value[0].upper()
        try:
            q = self.state.quoteState[part]
            if q.bidSize and q.askSize:
                value = (q.bid + q.ask) / 2
            else:
                value = q.last if q.last == q.last else q.close
        except:
            logger.error("[{}] No value found! Calculation can't continue.", value)
            return None

        logger.info("[{}] Using value: {:,.6f}", part, value)
        return Decimal(value)

    def number(self, value):
        """ur a number lol"""
        return Decimal(value[0].replace("_", ""))

    def operation(self, exprs):
        """Map from grammar symbol to running method"""
        # Get the operation symbol and translate to method name
        func_symbol = str(exprs[0])
        func_name = self.symbol_to_func.get(func_symbol.lower(), "")
        func = getattr(self, func_name, None)
        if func and callable(func):
            return func(exprs[1:])

    def add(self, numbers):
        """(+ a b c ...)"""
        return sum(numbers)

    def sub(self, numbers):
        """(- a b c ...)"""
        if len(numbers) == 1:
            return -numbers[0]

        return numbers[0] - sum(numbers[1:])

    def mul(self, numbers):
        """(* a b c ...)"""
        result = 1

        for number in numbers:
            result *= number

        return result

    def div(self, numbers):
        """(/ a b c ...)"""
        if numbers[1:] == 0:
            return Decimal("NaN")

        result = numbers[0]

        for number in numbers[1:]:
            result /= number
        return result

    def optgains(self, args):
        """(o qty contract-price)

        Just calculates a buy or sell price for a quantity and contract price (assuming 100 multiplier by default, but optional 3rd arg allows otheres)
        """
        qty, target, *mul = args

        # TODO: when using symbols for live quotes in calculations, thread the contract through the calculator so we can properly access contract.multiple here!
        multiplier = mul[0] if mul else 100

        # we're assuming 100x multiples. ymmv when estimating currencies or other futures.
        return qty * multiplier * target

    def gains(self, args):
        """(gains a b)

        generic percentage difference between a and b.
        percentage is provided as whole number (* 100) value.

        Exampe:
        100% gains because target 6 is twice as large as start 3:
            (gains 3 6) => 100

        50% loss because target 3 is half of 6:
            (gains 6 3) => -50
        """
        a, b = args
        return ((b - a) / a) * 100

    def grow(self, args):
        """(grow base percent duration)
        basically: (base * (1.percent)^duration)

        (grow 20000 6 20) => 64,142.7094
        """
        a, b, *c = args

        # if no duration provided, just do a single percentage growth multiply
        c = c[0] if c else 1

        return a * ((1 + (b / 100)) ** c)


@dataclass(slots=True)
class Calculator:
    state: Any

    parser: Lark = field(init=False)

    def __post_init__(self) -> None:
        self.parser = Lark(
            grammar, parser="lalr", transformer=CalculatorTransformer(self.state)
        )

    def calc(self, expression: str) -> Decimal:
        return self.parser.parse(expression).children[0]  # return result as string
