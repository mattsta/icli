"""A simple(ish) built-in calculator."""

from dataclasses import dataclass, field
from decimal import Decimal, getcontext

from lark import Lark, Transformer

getcontext().prec = 10  # set precision for Decimal

grammar = """
    start: expr
    ?expr: operation
         | NUMBER -> number
    operation: "(" FUNC expr+ ")"
    FUNC: "+" | "-" | "*" | "/" | "gains" | "grow" | "o"
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""


class CalculatorTransformer(Transformer):
    symbol_to_func = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "gains": "gains",
        "o": "optgains",
        "grow": "grow",
    }

    def number(self, value):
        """ur a number lol"""
        return Decimal(value[0])

    def operation(self, exprs):
        """Map from grammar symbol to running method"""
        # Get the operation symbol and translate to method name
        func_symbol = str(exprs[0])
        func_name = self.symbol_to_func.get(func_symbol, "")
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
        a, b, c = args
        return a * ((1 + (b / 100)) ** c)


@dataclass
class Calculator:
    parser: Lark = field(
        default_factory=lambda: Lark(
            grammar, parser="lalr", transformer=CalculatorTransformer()
        )
    )

    def calc(self, expression: str) -> Decimal:
        return self.parser.parse(expression).children[0]  # return result as string
