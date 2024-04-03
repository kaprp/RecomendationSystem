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
    brand: str = ""
    modelTitle: str = ""
    typeConnect: str = ""
    typeConstruction: str = ""
    mainColor: str = ""
    fasteningMethod: str = ""
    microphoneMount: str = ""
    typeAcousticDesign: str = ""
    typeCountEmitter: str = ""
    typeCharging: str = ""
    connectCable: str = ""
    ChargingUsb: bool = False
    codecs: str = ""
    radius: str = ""
    ipyz: str = ""
    typeSoundScheme: str = ""
    versionBluetooth: str = ""
    battery: str = ""
    weight: str = ""
    chargingCase: str = ""
    time: str = ""
    sense: str = ""
    freqRange: str = ""
    diametr: str = ""
    microphone: bool = False
    volumeControl: bool = False
    typeVolControl: str = ""
    url: str = ""
    price: str = ""
    countRate: str = ""
    averageRate: str = ""
    typeAcousticType: str = ""
    typeCon: str = ""
