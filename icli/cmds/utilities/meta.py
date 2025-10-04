"""Command: meta

Category: Utilities
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import _COMMAND_REGISTRY, IOp, command
from icli.helpers import *

if TYPE_CHECKING:
    pass
import inspect

import pandas as pd


@command(names=["meta"])
@dataclass
class IOpMeta(IOp):
    """Show internal command system state and debug information.

    Usage:
        meta [mode]

    Modes:
        summary  - Show all commands with categories and aliases (default)
        detail   - Show detailed view with source files and descriptions
        files    - Group commands by their source file
        stats    - Show command system statistics and counts
        all      - Show all views combined

    Examples:
        meta              # Shows summary view
        meta stats        # Shows statistics only
        meta detail       # Shows detailed command listing
        meta all          # Shows everything
    """

    mode: str = "summary"

    def argmap(self):
        return [
            DArg(
                "mode",
                default="summary",
                desc="Display mode: 'summary' (default), 'detail', 'files', or 'stats'",
            )
        ]

    async def run(self):
        mode = self.mode.lower()

        if mode in {"summary", "detail", "details", "all"}:
            await self.show_command_listing(
                detailed=mode in {"detail", "details", "all"}
            )

        if mode in {"files", "all"}:
            await self.show_files_listing()

        if mode in {"stats", "all"}:
            await self.show_statistics()

    async def show_command_listing(self, detailed: bool = False):
        """Show all commands organized by category with their source files."""
        rows = []

        for cmd_class in sorted(
            _COMMAND_REGISTRY, key=lambda c: (c.__command_category__, c.__name__)
        ):  # type: ignore[attr-defined]
            category = cmd_class.__command_category__  # type: ignore[attr-defined]
            names = cmd_class.__command_names__  # type: ignore[attr-defined]

            # Get the source file path
            try:
                source_file_obj = inspect.getfile(cmd_class)
                # Make it relative to the project root for readability
                import pathlib

                source_file = str(
                    pathlib.Path(source_file_obj).relative_to(pathlib.Path.cwd())
                )
            except:
                source_file = "unknown"

            # Get docstring
            doc = (
                (cmd_class.__doc__ or "").strip().split("\n")[0]
                if cmd_class.__doc__
                else ""
            )

            primary_name = names[0]
            aliases = ", ".join(names[1:]) if len(names) > 1 else ""

            row = {
                "Category": category,
                "Command": primary_name,
                "Aliases": aliases,
                "Class": cmd_class.__name__,
            }

            if detailed:
                row["File"] = str(source_file)
                row["Description"] = doc[:80] + "..." if len(doc) > 80 else doc

            rows.append(row)

        df = pd.DataFrame(rows)

        title = (
            "Command System Overview"
            if not detailed
            else "Command System Detailed View"
        )
        printFrame(df, title)

    async def show_files_listing(self):
        """Show commands grouped by source file."""
        files_map: dict[str, list[dict[str, str]]] = {}

        for cmd_class in _COMMAND_REGISTRY:
            try:
                source_file_obj = inspect.getfile(cmd_class)
                import pathlib

                source_file = str(
                    pathlib.Path(source_file_obj).relative_to(pathlib.Path.cwd())
                )
            except:
                source_file = "unknown"

            if source_file not in files_map:
                files_map[source_file] = []

            files_map[source_file].append(
                {
                    "Command": ", ".join(cmd_class.__command_names__),  # type: ignore[attr-defined]
                    "Class": cmd_class.__name__,
                    "Category": cmd_class.__command_category__,  # type: ignore[attr-defined]
                }
            )

        rows = []
        for file_path in sorted(files_map.keys()):
            for cmd_info in files_map[file_path]:
                rows.append({"File": file_path, **cmd_info})

        df = pd.DataFrame(rows)
        printFrame(df, "Commands by Source File")

    async def show_statistics(self):
        """Show command system statistics."""
        from icli import cmds

        # Category statistics
        category_counts = {}
        total_commands = 0
        total_aliases = 0

        for category, cmd_map in cmds.OP_MAP.items():
            category_counts[category] = len(cmd_map)
            total_commands += len(cmd_map)

        # Count unique classes vs total command names (to show aliases)
        unique_classes = len(_COMMAND_REGISTRY)

        for cmd_class in _COMMAND_REGISTRY:
            total_aliases += len(cmd_class.__command_names__) - 1  # type: ignore[attr-defined]

        logger.info("=" * 60)
        logger.info("COMMAND SYSTEM STATISTICS")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Total Categories:       {}", len(cmds.OP_MAP))
        logger.info("Total Commands:         {}", total_commands)
        logger.info("Unique Command Classes: {}", unique_classes)
        logger.info("Total Command Aliases:  {}", total_aliases)
        logger.info("")
        logger.info("Commands by Category:")
        logger.info("-" * 60)

        for category in sorted(category_counts.keys()):
            count = category_counts[category]
            logger.info("  {:30s} {:>3d} commands", category, count)

        logger.info("")
        logger.info("=" * 60)

        # Also show as a dataframe
        rows = [
            {"Category": cat, "Commands": count}
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])
        ]

        # Add totals row
        rows.append({"Category": "TOTAL", "Commands": total_commands})

        df = pd.DataFrame(rows)
        printFrame(df, "Commands per Category")
