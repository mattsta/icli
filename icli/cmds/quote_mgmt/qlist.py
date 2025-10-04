"""Command: qlist

Category: Quote Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import prettyprinter as pp  # type: ignore


@command(names=["qlist"])
@dataclass
class IOpQuoteList(IOp):
    """Show current quote group names usable by other q* commands"""

    groups: set[str] = field(init=False)

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
            groups = sorted([k[1] for k in self.cache if k[0] == "quotes"])  # type: ignore

        logger.info("Groups: {}", pp.pformat(groups))

        for group in groups:
            found = self.cache.get(("quotes", group), [])  # type: ignore

            if isinstance(found, list):
                found = sorted(found)

            # printing all contract IDs directly can make it easy to copy all current quotes
            # to another system or sharing a quote view with people who don't have your saved
            # quote database.
            if isinstance(found, dict) and "contracts" in found:
                contracts = found["contracts"]
                contracts = sorted(contracts, key=lambda x: x.localSymbol)

                logger.info(
                    "[{}] Members (conIds): {}",
                    group,
                    " ".join([str(x.conId) for x in contracts]),
                )

            logger.info("[{}] Members: {}", group, pp.pformat(found))
