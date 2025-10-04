"""Command: ifgroup

Category: Predicate Management
"""

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg
from tradeapis import ifthen, ifthen_dsl

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass


@command(names=["ifgroup"])
@dataclass
class IOpPredicateGroup(IOp):
    """Manage a predicate group (or tree) structure embedded in one command.

      Purpose: Allow in-cli creation of OCA and OTO and OTOCO if/then trees.

      Basically we need to:
        - create various named ifthen predicates
        - attach named predicates to either OTO (active -> waiting) or OCA (peers) groups
        - we can also attach OTO or OCA groups to other OTO or OCA groups as well for continually recursive soultions (e.g. short -> long -> short -> long -> ...)

    Perhaps the simplest approach here is to buffer all the live descriptions then build the ConfigLoader yaml for processing?
    Alternatively, just provide a config loader yaml file itself for injection into the current cli session.

    NOTE: This is probably completely obsoleted by the `auto` IOpPredicateAutoRunner loader and named ifthen dsl predicate management system.
    """

    name: bool = field(init=False)
    predicate: str = field(init=False)

    def argmap(self):
        return [
            DArg(
                "name",
                desc="Name of this predicate for attaching to other places (can also be a special override command)",
            ),
            DArg(
                "*predicate",
                convert=lambda x: " ".join(x),
                desc="Content of predicate and command to execute upon completion",
            ),
        ]

    async def run(self):
        match self.name:
            case ":load":
                doit = pathlib.Path(self.predicate)
                logger.info("Loading predicate config file: {}", doit)
                match doit.suffix:
                    case ".yaml" | ".yml":
                        loader = ifthen.IfThenConfigLoader(self.state.ifthenRuntime)
                    case ".ifthen":
                        loader = ifthen_dsl.IfThenDSLLoader(self.state.ifthenRuntime)
                    case _:
                        logger.error(
                            "[{}] Filename must end in .yaml, .yml, or .ifthen depending on flow control syntax.",
                            doit,
                        )
                        return

                content = doit.read_text()
                logger.info("[{}] Generate predicates from:\n{}", doit, content)
                created, starts, populate = loader.load(content, activate=False)

                # after loaded/created/activated, we now must POPULATE each predicate with our custom data function attachments
                for pid in populate:
                    await self.state.predicateSetup(self.state.ifthenRuntime[pid])

                loader.activate(starts)

                logger.info("[{}] Created {} predicates!", doit, created)
            case _:
                logger.info("[{}] -> {}", self.name, self.predicate)
