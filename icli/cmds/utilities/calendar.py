"""Command: calendar

Category: Utilities
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import calendar

import whenever


@command(names=["calendar"])
@dataclass
class IOpCalendar(IOp):
    """Just show a calendar!"""

    year: list[str] | None = field(init=False)

    def argmap(self):
        return [
            DArg(
                "*year",
                desc="Year for your calendar to show (if not provided, just use current year)",
            )
        ]

    async def run(self):
        try:
            assert self.year
            year = int(self.year[0])
        except:
            year = whenever.ZonedDateTime.now("US/Eastern").year

        # MURICA
        # (also lol for this outdated python API where you have to globally set the calendar start
        #  date for your entire environment!)
        calendar.setfirstweekday(calendar.SUNDAY)
        logger.info("[{}] Calendar:\n{}", year, calendar.calendar(year, 1, 1, 6, 3))
