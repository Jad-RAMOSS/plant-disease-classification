from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    data_dir: str = 'plantvillage dataset/color'
    save_path: str = 'outputs'
    subject: str = 'Plant Village Disease'

    img_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    batch_size: int = 40

    freeze_base: bool = False
    learning_rate: float = 0.001
    epochs: int = 2
    patience: int = 1
    stop_patience: int = 3
    threshold: float = 0.9
    factor: float = 0.5
    ask_epoch: int = 5

    @property
    def img_shape(self) -> Tuple[int, int, int]:
        return (self.img_size[0], self.img_size[1], self.channels)
