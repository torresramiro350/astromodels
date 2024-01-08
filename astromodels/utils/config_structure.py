import logging
from dataclasses import dataclass, field
from enum import IntEnum, Enum


# logging
class LoggingLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class Logging:
    path: str = "~/.astromodels/log"
    developer: bool = "off"
    usr: bool = "on"
    console: bool = "on"
    level: LoggingLevel = LoggingLevel.INFO
    startup_warnings: bool = "on"
    info_style: str = "medium_spring_green"
    warn_style: str = "medium_orchid"
    error_style: str = "blink bold bright_red"
    debug_style: str = "blue_violet"
    message_style: str = "bold grey78"


class AbsTables(Enum):
    WILM = "WILM"
    ASPL = "ASPL"
    AG89 = "AG89"


class EBLTable(Enum):
    franceschini = "franceschini"
    kneiske = "kneiske"
    dominguez = "dominguez"
    inuoe = "inuoe"
    gilmore = "gilmore"


@dataclass
class AbsorptionModels:
    tbabs_table: AbsTables = AbsTables.WILM
    phabs_table: AbsTables = AbsTables.AG89
    ebl_table: EBLTable = EBLTable.dominguez


@dataclass
class Modeling:
    use_memoization: bool = True
    use_parameter_transforms: bool = True
    ignore_parameter_bounds: bool = False


@dataclass
class Config:
    logging: Logging = field(default_factory=Logging)
    absorption_models: AbsorptionModels = field(
        default_factory=AbsorptionModels
    )
    modeling: Modeling = field(default_factory=Modeling)
