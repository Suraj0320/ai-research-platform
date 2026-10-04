from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def run(self, task: str) -> str:
        """
        Execute the agent for the given task.
        """
        pass