from enum import Enum

# TODO: Refactor into polymorphic classes
class Channel(Enum):
    LASER_POSITION = 1
    TEMPERATURE = 2
    TEMPERATURE_AROUND_LASER = 3


# TODO: Refactor into polymorphic classes
class ExtraParam(Enum):
    SCANNING_ANGLE = 1
    LASER_POWER = 2
    LASER_RADIUS = 3