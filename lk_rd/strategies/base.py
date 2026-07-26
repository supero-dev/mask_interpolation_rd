from abc import ABC, abstractmethod


class PropagationStrategy(ABC):
    name = "base"

    def on_anchor(self, frame_id, prediction):
        pass

    @abstractmethod
    def propagate(self, prev, prev_gray, gray, velocity):
        raise NotImplementedError
