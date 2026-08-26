from .base import TrainingSample
from .metatft import MetaTFTSource
from .opgg import OpggLiveSource
from .metatft_pro import MetaTFTProSource
from .riot import RiotHighEloSource
from .webstats import LolchessSource, TacticsToolsSource

__all__ = [
    "TrainingSample",
    "MetaTFTSource",
    "OpggLiveSource",
    "MetaTFTProSource",
    "RiotHighEloSource",
    "LolchessSource",
    "TacticsToolsSource",
]
