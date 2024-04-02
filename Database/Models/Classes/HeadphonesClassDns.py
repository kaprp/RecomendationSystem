from dataclasses import dataclass, field
from typing import List

@dataclass
class IPYZ:
    ip: str = "IP"
    Y: str = "Y"
    Z: str = "Z"

    def __get_item__(self):
        return self.ip+self.Y+self.Z


@dataclass
class Headphones:
    name: str = ""
    modelTitle: str = ""
    typeModel: str = ""
    country: str = ""
    typeConnect: str = ""
    typeConstruction: str = ""
    mainColor: str = ""
    materialHousing: str = ""
    materialEarPads: str = ""
    fasteningMethod: str = ""
    microphoneLocation: str = ""
    microphoneMount: str = ""
    price: str = ""
    typeAcousticDesign: str = ""
    typeCountEmitter: str = ""
    typeCharging: str = ""
    url: str = ""
    maker: str = ""

    # images: list = field(default_factory=list)
    # codecs: list = field(default_factory=list)
    # functionsKeys: list = field(default_factory=list)
    # functionalFeatures: list = field(default_factory=list)
    # equipment: list = field(default_factory=list)
    # otherFeatures: list = field(default_factory=list)
    # reviews: list = field(default_factory=list)
    codecs: str = ""
    functionsKeys: str = ""
    functionalFeatures: str = ""
    equipment: str = ""
    otherFeatures: str = ""
    reviews: str = ""

    countEmitter: int = 0
    assurance: int = 0
    countRate: int = 0
    # Степень пылевлагозашиты
    ipyz = IPYZ().__get_item__()
    typeSoundScheme: float = 0.0
    versionBluetooth: float = 0.0
    battery: float = 0.0
    weight: float = 0.0
    chargingCase: float = 0.0
    lengthCable: float = 0.0
    averageRate: float = 0.0
    impendance: float = 0.0

    microphone: bool = False
    gamingType: bool = False
    detachableMicrophone: bool = False
    backlight: bool = False
    childDesign: bool = False
    sportFactor: bool = False
    cableCharging: bool = False
    activeNoiseReduction: bool = False
    transparentMode: bool = False
    touchControl: bool = False
    volumeControl: bool = False
    controlSmartphone: bool = False
    fastCharging: bool = False


@dataclass
class HeadphonesCit(Headphones):
    # properties: list = field(default_factory=list)
    properties: str = ""