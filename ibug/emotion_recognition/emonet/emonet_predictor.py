import os
import torch
import numpy as np
from types import SimpleNamespace
from typing import Union, Optional, Dict
from .emonet import EmoNet


__all__ = ['EmoNetPredictor']


class EmoNetPredictor(object):
    def __init__(self, device: Union[str, torch.device] = 'cuda:0', model: Optional[SimpleNamespace] = None) -> None:
        self.device = device
        if model is None:
            model = EmoNetPredictor.get_model()
        self.config = model.config
        self.net = EmoNet(config=self.config).to(self.device)
        self.net.load_state_dict(torch.load(model.weights, map_location=self.device))
        self.net.eval()
        if self.config.use_jit:
            self.net = torch.jit.trace(self.net, torch.rand(
                1, self.config.num_input_channels, self.config.input_size, self.config.input_size).to(self.device))
        pass

    @staticmethod
    def get_model(name: str = 'emonet248') -> SimpleNamespace:
        name = name.lower()
        if name == 'emonet248':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'emonet248.pth'),
                                   num_input_channels=768, input_size=64, n_blocks=4, n_expression=8, n_reg=2)
        elif name == 'emonet245':
            return SimpleNamespace(weights=os.path.join(os.path.dirname(__file__), 'weights', 'emonet245.pth'),
                                   num_input_channels=768, input_size=64, n_blocks=4, n_expression=5, n_reg=2)
        else:
            raise ValueError("name must be set to either emonet248 or emonet245")

    @torch.no_grad()
    def __call__(self, fan_features: torch.Tensor) -> Dict:
        result = self.net(fan_features.to(self.device)).cpu().numpy()
        return {'expression': np.argmax(result[:, :-2], axis=1),
                'valence': result[:, :-2], 'arousal': result[:, :-1],
                'raw_result': result}
