"""Command: limit

Category: Order Management
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from ib_async import (
    Contract,
    PortfolioItem,
)
from loguru import logger
from mutil.dispatch import DArg
from questionary import Choice

from icli.cmds.base import IOp, command
from icli.cmds.utils import (
    ORDER_TYPE_Q,
)
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["limit"])
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
