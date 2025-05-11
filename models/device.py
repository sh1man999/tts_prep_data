from enum import Enum


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"