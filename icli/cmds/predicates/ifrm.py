"""Command: ifrm

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


@command(names=["ifrm"])
@dataclass
class IOpPredicateDelete(IOp):
    """Delete one or more predicates by their current session id (visible at creation time or from iflist)."""

    ids: set[int] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*ids",
                convert=lambda x: set(map(int, x)),
                desc="Predicate IDs to delete (from iflist)",
            )
        ]

    async def run(self):
        logger.info("Deleting Predicates: {}", sorted(self.ids))
        for pid in self.ids:
            try:
                predicate = self.state.ifthenRuntime.remove(pid)
                logger.info("[{}] predicate deleted: {}", pid, pp.pformat(predicate))
            except:
                logger.warning("[{}] predicate not found; nothing deleted", pid)
