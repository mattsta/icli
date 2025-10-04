"""Command: auto

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
import asyncio

import prettyprinter as pp  # type: ignore


@command(names=["auto"])
@dataclass
class IOpPredicateAutoRunner(IOp):
    """Create/enable or stop named predicate DSL configurations.

    Purpose: Populate content of pre-existing predicate complate configs with live configuration
             data then run/enable the predicate to operate using live market data and live account access.

    Implementation note: This is a refactored/second-generation version of the `ifgroup :load` system where, instead of only loading
                         from static files, the files can be templates which we live-populate at runtime (and the system can have some
                         built-in templates or operate from self-assembled template strings directly instead of needing everything to
                         be from actual files).

    Usage like:
      auto start builtin:algo_flipper.dsl spy-flipper watch_symbol=SPY algo_symbol=SPY evict_symbol=SPY trade_symbol=SPY qty=500 profit_pts=2 loss_pts=2
      auto stop spy-long
      auto report spy-long

    Basically we need to:
      - create predicate config from IfThenRuntimeTemplateExecutor
      - populate user-provided variables/config into the predicate (symbol details, algo, duration, qty, profit/loss targets)
      - start the predicate
      - when trades execute, potentially update the template executor state for success/failure reporting (requires more structure around _what_ is triggering each order and mapping order ids back to "order originiating owners")
    """

    action: str = field(init=False)
    templateName: str = field(init=False)
    name: str = field(init=False)
    properties: dict[str, str] = field(init=False)

    def argmap(self):
        return [
            DArg(
                "action",
                convert=lambda x: x.lower(),
                verify=lambda x: x
                in {"start", "stop", "preview", "report-all", "report"},
                desc="Command for predicate runner. one of: start, stop, report",
            ),
            DArg(
                "templateName",
                verify=lambda x: ":" in x
                and x.split(":")[0].lower() in {"builtin", "file"},
                desc="Name of template to pull for applying replacements and creating predicates. Note: template format is kind:name where kind is `builtin` or `file`",
            ),
            DArg(
                "name",
                desc="Name for template+parameters combination (scoped to `templateName` for all operations)",
            ),
            DArg(
                "*properties",
                # *properties format is: ["algo_symbol=SPY", "qty=50", ...]
                # and we want to convert it to a key-val dict, so... split on each '=' then take k/v pairs
                convert=lambda x: dict([kv.strip().split("=", 1) for kv in x]),
                desc="key-value parameters used in template",
            ),
        ]

    def parseProperties(self):
        """Replace any positional markers like :N in param values with the contract names."""
        for k, v in self.properties.items():
            if ":" in v:
                # capture :N values
                # resolve :N values to underlying symbol name
                # replace resolved value in original string for each occurrance
                self.properties[k] = self.state.scanStringReplacePositionsWithSymbols(v)

        return self.properties

    async def run(self):
        it = self.state.ifthenTemplates

        def loadTemplate():
            # Template names must be namespaced so we can look them up correctly.
            # Either: builtin:name or file:name
            frm, tn = self.templateName.lower().split(":")

            # replace any ":N" references in the template parameters with symbol names (where useful)
            self.parseProperties()

            # Running the creation is idempotent if the content between names and targets are the same between calls.
            match frm:
                # Note: 'templateName' is the, well, template name, but each _instance_ of a template is self.name
                case "builtin":
                    it.from_builtin(tn, self.templateName)
                case "file":
                    it.from_file(tn, self.templateName)
                case _:
                    raise ValueError(
                        f"templateName arg must have namespace builtin: or file: - given {self.templateName}"
                    )

        match self.action:
            case "preview":
                loadTemplate()
                t = it.preview_template(self.templateName, self.properties)
                logger.info("[{}] {}", self.templateName, t)
            case "start":
                loadTemplate()
                # populate template with name and args
                # TODO: debug why we need enable=True here _and_ manual .activate() at the end,
                #       since 'enable=True' here should cause the runtime to genreate an activation itself?
                #       We lost track of how the activations work somewhere along the way and which ones
                #       are enabling which features.
                created_count, start_ids, all_ids = it.activate(
                    self.templateName, self.name, self.properties, enable=True
                )

                # this is just a silly way of telling mypy we do not have any `None` entries in
                # the resolved predicates we are about to call .predicateSetup() on for all entries.
                crs = filter(None, [self.state.ifthenRuntime[pid] for pid in all_ids])

                # attach data handlers to all individual predicate ids needing data updates
                await asyncio.gather(*[self.state.predicateSetup(c) for c in crs])

                # start the top-level predicate entry points
                for start_id in start_ids:
                    logger.info("Starting ifthen predicate id: {}", start_id)
                    self.state.ifthenRuntime.activate(start_id)
            case "stop":
                it.deactivate(self.templateName, self.name)
            case "report-all":
                logger.info(
                    "{}",
                    pp.pformat(it.get_system_health_report()),
                )
            case "report":
                logger.info(
                    "[{} :: {}] :: {}",
                    self.templateName,
                    self.name,
                    pp.pformat(
                        it.get_performance_summary(self.templateName, self.name)
                    ),
                )
                logger.info(
                    "[{} :: {}] :: {}",
                    self.templateName,
                    self.name,
                    pp.pformat(it.get_performance_events(self.templateName, self.name)),
                )
            case _:
                logger.error(
                    "[{} :: {}] :: Unknown command?", self.templateName, self.name
                )
