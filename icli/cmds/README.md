# Command System Documentation

The command system has been refactored from a single monolithic `lang.py` file (7,589 lines) into a modular, plugin-like architecture.

## Architecture Overview

### Directory Structure

```
icli/cmds/
├── __init__.py           # Auto-discovery & OP_MAP builder
├── base.py               # IOp base class + @command decorator
├── dispatch.py           # Dispatch class for routing commands
├── utils.py              # Shared utilities from lang.py
├── quotes/               # Live Market Quotes (10 commands)
├── orders/               # Order Management (8 commands)
├── predicates/           # Predicate Management (7 commands)
├── portfolio/            # Portfolio (6 commands)
├── connection/           # Connection (1 command)
├── utilities/            # Utilities (19 commands)
├── schedule/             # Schedule Management (6 commands)
├── tasks/                # Task Management (2 commands)
└── quote_mgmt/           # Quote Management (12 commands)
```

**Total:** 64 command files across 9 categories, 71 total command names (including aliases)

## How It Works

### 1. Command Registration

Commands register themselves using the `@command` decorator. The category is **automatically detected** from the directory structure - no redundancy!

```python
from icli.cmds.base import IOp, command

@command(names=["positions", "ls"])  # Category auto-detected from directory!
@dataclass
class IOpPositions(IOp):
    def argmap(self):
        return [DArg("*symbols")]

    async def run(self):
        # Implementation
```

Each category directory defines its display name once in `__init__.py`:
```python
# icli/cmds/portfolio/__init__.py
"""Portfolio commands."""
CATEGORY = "Portfolio"
```

### 2. Auto-Discovery

When `from icli import cmds` is imported:

1. `cmds/__init__.py` scans all subdirectories
2. Imports all `.py` files (except infrastructure files)
3. Each `@command` decorated class registers itself in `_COMMAND_REGISTRY`
4. `OP_MAP` is built from the registry: `{category: {name: CommandClass}}`

### 3. Command Execution

The `Dispatch` class uses `OP_MAP` to route commands:

```python
from icli import cmds

dispatch = cmds.Dispatch()
await dispatch.runop("positions", "SPY AAPL", state)
```

## Adding New Commands

### Option 1: Add to Existing Category

Create a new file in the appropriate category directory:

```python
# icli/cmds/portfolio/summary.py

from dataclasses import dataclass
from icli.cmds.base import IOp, command
from icli.helpers import *
from mutil.dispatch import DArg

@command(names=["summary"])  # Category auto-detected!
@dataclass
class IOpPortfolioSummary(IOp):
    def argmap(self):
        return []

    async def run(self):
        # Your implementation
```

The command is automatically discovered on next import!

### Option 2: Create New Category

1. Create directory: `icli/cmds/my_category/`
2. Add `__init__.py`:
   ```python
   """My Category commands."""
   CATEGORY = "My Category"
   ```
3. Add command files (category auto-detected from directory!)

## Key Features

✅ **Auto-Discovery** - Drop in a command file, it's automatically registered
✅ **Auto-Category Detection** - No redundant category specifications in decorators
✅ **Duplicate Detection** - Prevents conflicting command names at startup
✅ **Clean Separation** - Each command in its own file
✅ **Type Safety** - Full type hints and mypy support
✅ **Shared Utilities** - Common functions in `utils.py`
✅ **Command Aliases** - Multiple names for same command (e.g., "ls" and "positions")
✅ **Category Organization** - Commands grouped by function
✅ **Single Point of Responsibility** - Category defined once per directory

## Infrastructure Files

- **base.py**: `IOp` base class and `@command` decorator
- **dispatch.py**: `Dispatch` class for command routing
- **utils.py**: Shared utilities (expand_symbols, automaticLimitBuffer, etc.)
- **__init__.py**: Discovery system and OP_MAP builder

## Migration Notes

The refactoring was performed using an AST-based extraction tool (`tools/extract_commands.py`) that:

1. Parsed the original `lang.py` using Python's AST module
2. Extracted `OP_MAP` to determine command->category mappings
3. Identified all `@dataclass` classes inheriting from `IOp`
4. Analyzed imports needed by each command
5. Generated individual command files with proper imports
6. Extracted shared utilities to `utils.py`

The old `lang.py` has been archived as `lang.py.old` for reference.

## Testing

To verify the command system:

```bash
# Test import
poetry run python -c "from icli import cmds; print(len(cmds.OP_MAP), 'categories loaded')"

# Test dispatch
poetry run python -c "from icli import cmds; d = cmds.Dispatch(); print('Dispatch OK')"
```

## Command Categories

| Category | Commands | Description |
|----------|----------|-------------|
| Live Market Quotes | 10 | Quote requests, depth, chains, prequalify |
| Order Management | 8 | Buy, limit, scale, modify, cancel orders |
| Predicate Management | 7 | ifthen rules and automation |
| Portfolio | 6 | Positions, orders, balance, executions |
| Connection | 1 | Connection management (rid) |
| Utilities | 19 | Misc tools (cash, calendar, math, etc.) |
| Schedule Management | 6 | Schedule events (sadd, slist, scancel) |
| Task Management | 2 | Background tasks (tasklist, taskcancel) |
| Quote Management | 12 | Quote save/restore/cleanup operations |

---

*Generated during command system refactoring - 2025*
