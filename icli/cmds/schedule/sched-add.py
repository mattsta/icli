"""Command: sched-add, sadd

Category: Schedule Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.bgtask import BGSchedule
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import asyncio

import dateutil.parser
import pytz
import whenever


@command(names=["sched-add", "sadd"])
@dataclass
class IOpScheduleEvent(IOp):
    """Schedule a command to execute at a specific date+time in the future."""

    name: str = field(init=False)
    datetime: whenever.ZonedDateTime = field(init=False)
    cmd: list[str] = field(init=False)

    # asub /NQ COMBONQ yes 0.66 cash 15 TemaTHMAFasterSlower direct
    def argmap(self):
        return [
            DArg(
                "name",
                desc="Name of event (for listing and canceling in the future if needed)",
            ),
            DArg(
                "datetime",
                convert=lambda dt: whenever.LocalDateTime.from_py_datetime(
                    dateutil.parser.parse(dt)
                ).assume_tz("US/Eastern", disambiguate="compatible"),
                desc="Date and Time of event (timezone will be Eastern Time)",
            ),
            DArg("cmd", desc="icli command to run at the given time"),
        ]

    async def run(self):
        if self.name in self.state.scheduler:
            logger.error(
                "[{} :: {}] Can't schedule because name already scheduled!",
                self.name,
                self.cmd,
            )
            return False

        now = whenever.ZonedDateTime.now("US/Eastern")

        # "- 1 second" allows us to schedule for "now" without time slipping into the past and
        # complaining we scheduled into the past. sometimes we just want it now.
        if (now - whenever.seconds(1)) > self.datetime:
            logger.error(
                "You requested to schedule something in the past? Not scheduling."
            )
            return False

        logger.info(
            "[{} :: {} :: {}] Scheduling: {}",
            self.name,
            self.datetime,
            dict(
                zip(
                    "hours minutes seconds".split(),
                    (self.datetime - now).in_hrs_mins_secs_nanos(),
                )
            ),
            self.cmd,
        )

        async def doit() -> None:
            try:
                # "self.cmd" is the text format of a command prompt to run.
                # You can run multiple commands with standard "ls; orders; exec" syntax
                logger.info("[{} :: {}] RUNNING UR CMD!", self.name, self.cmd)
                await self.state.buildAndRun(self.cmd)
                logger.info("[{} :: {}] Completed UR CMD!", self.name, self.cmd)
            except asyncio.CancelledError:
                logger.warning(
                    "[{} :: {}] Future Scheduled Task Canceled!", self.name, self.cmd
                )
            except:
                logger.exception(
                    "[{} :: {}] Scheduled event failed?", self.name, self.cmd
                )
            finally:
                self.state.scheduler.cancel(self.name)
                logger.info("[{}] Removed scheduled event!", self.name)

        howlong = (self.datetime - now).in_seconds()
        logger.info(
            "[{} :: {}] command is scheduled to run in {:,.2f} seconds ({:,.2f} minutes)!",
            self.name,
            self.cmd,
            howlong,
            howlong / 60,
        )

        sched = self.state.scheduler.create(
            self.name,
            doit(),
            schedule=BGSchedule(
                start=self.datetime.py_datetime(), tz=pytz.timezone("US/Eastern")
            ),
            meta=self.cmd,
        )
        logger.info("[{} :: {}] Scheduled via: {}", self.name, self.cmd, sched)
