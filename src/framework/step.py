from abc import ABC, abstractmethod
import os

class Step(ABC):
    """
    Abstract base class for all workflow steps.
    
    Provides default implementation for get_log_dir that supports appliance_name.
    """
    def __init__(self, name: str):
        self.name = name
        self.appliance_name = ""  # Default empty, can be set by subclasses

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        Execute the step logic.
        
        Args:
            context (dict): Shared dictionary containing sequence_id, paths, and intermediate data.
            
        Returns:
            dict: Updated context.
        """
        pass

    def restore(self, context: dict) -> dict:
        return context

    def get_log_dir(self, context: dict) -> str:
        """
        Get the specific log directory for this step, optionally with appliance name.
        
        Supports appliance_name for organizing logs by device type:
        - With appliance_name: log/{appliance_name}_{sequence_id}/{step_name}/
        - Without appliance_name: log/{sequence_id}/{step_name}/
        
        Args:
            context (dict): Shared context containing sequence_id and log_root
            
        Returns:
            str: Path to step's log directory
        """
        log_root = context.get('log_root', os.path.join('log', 'default'))
        step_log_dir = os.path.join(log_root, self.name)
        os.makedirs(step_log_dir, exist_ok=True)
        return step_log_dir
