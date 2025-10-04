"""Command: qdelete

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


@command(names=["qdelete"])
@dataclass
class IOpQuoteGroupDelete(IOp):
    """Delete an entire quote group"""

    groups: set[str] = field(init=False)

    def argmap(self):
        return [DArg("*groups", convert=set, desc="quote group names to delete.")]

    async def run(self):
        if not self.groups:
            logger.error("No groups provided!")

        for group in sorted(self.groups):
            logger.info("Deleting quote group: {}", group)
            self.cache.delete(("quotes", group))  # type: ignore[attr-defined]
