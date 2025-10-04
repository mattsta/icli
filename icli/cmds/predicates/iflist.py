"""Command: iflist, ifls

Category: Predicate Management
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


@command(names=["iflist", "ifls"])
@dataclass
class IOpPredicateList(IOp):
    """List all actively running predicates with their runtime IDs"""

    pids: list[int] = field(init=False)

    def argmap(self):
        return [DArg("*pids", convert=lambda x: [int(y) for y in x])]

    async def run(self):
        preds, actives = self.state.ifthenRuntime.report()

        if self.pids:
            for p in self.pids:
                logger.info(
                    "[{}] Found ({}): {}",
                    p,
                    "ACTIVE" if p in actives else "NOT ACTIVE",
                    pp.pformat(preds.get(p)),
                )
        else:
            logger.info("Active Predicates ({}): {}", len(actives), sorted(actives))
            logger.info("All Predicates ({}):", len(preds))
            logger.info("{}", pp.pformat(preds))
