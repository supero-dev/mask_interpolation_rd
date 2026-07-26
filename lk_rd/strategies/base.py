from abc import ABC, abstractmethod


class PropagationStrategy(ABC):
    name = "base"

    @abstractmethod
    def propagate(self, prev, prev_gray, gray, velocity):
        raise NotImplementedError

