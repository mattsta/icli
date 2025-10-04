"""Command: ifthen

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


@command(names=["ifthen"])
@dataclass
class IOpPredicateCreate(IOp):
    predicate: str = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*predicate",
                convert=lambda x: " ".join(x).strip(),
                verify=lambda x: bool(x) and len(x) > 10,
            )
        ]

    async def run(self):
        logger.info("[{}] Configuring predicate...", self.predicate)

        # We need to:
        #   - record predicate with target operations into state we can read/delete
        #   - attach live price/algo fetcher to predicate so it can check itself on every ticker update
        #   - subscribe predicate to symbols used for decision making
        #   - view current state of each element of the predicate

        pid = self.state.ifthenRuntime.parse(self.predicate)

        prepredicate = self.state.ifthenRuntime[pid]
        logger.info("[{}] Parsed: {}", pid, pp.pformat(prepredicate))

        assert prepredicate is not None

        await self.state.predicateSetup(prepredicate)

        # now, since we attached the proper data extractors, we can enable the predicate for running
        # Here, for SINGLE predicates, we mark them to delete after one success.
        self.state.ifthenRuntime.activate(pid, once=True)
