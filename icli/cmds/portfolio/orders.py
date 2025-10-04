"""Command: orders

Category: Portfolio
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ib_async import (
    Bag,
    Contract,
    FuturesOption,
    Option,
)
from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.cmds.utils import (
    addRowSafe,
)
from icli.helpers import *

if TYPE_CHECKING:
    pass
import dateutil.parser
import pandas as pd
import prettyprinter as pp  # type: ignore


@command(names=["orders"])
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
