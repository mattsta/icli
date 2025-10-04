"""Command: cancel

Category: Order Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg
from questionary import Choice

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import fnmatch


@command(names=["cancel"])
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
