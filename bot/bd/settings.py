from dataclasses import dataclass

@dataclass
class Settings:
    use_link: bool = False
    markets_link: str = "DNS"
    viewed_marketplaces: str = "DNS"
    


