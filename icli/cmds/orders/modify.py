"""Command: modify

Category: Order Management
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from ib_async import Order
from loguru import logger
from mutil.numeric import fmtPrice
from questionary import Choice

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import prettyprinter as pp  # type: ignore


@command(names=["modify"])
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
