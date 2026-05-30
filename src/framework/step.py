from abc import ABC, abstractmethod
import os

class Step(ABC):
    """
    Abstract base class for all workflow steps.
    
    Provides default implementation for get_log_dir that supports appliance_name.
    """
    def __init__(self, name: str, suffix: str = ""):
        self.name = name
        self.suffix = suffix
        self.appliance_name = ""  # Default empty, can be set by subclasses
        self.save_interval = 0

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

    def should_save_intermediate(self, count: int, context: dict) -> bool:
        """
        Check if intermediate results should be saved based on the count and context['save_interval'].
        """
        interval = context.get('save_interval', 0)
        if interval > 0 and count > 0 and count % interval == 0:
            return True
        return False

    def restore(self, context: dict) -> dict:
        return context

    def get_log_dir(self, context: dict) -> str:
        """
        Get the specific log directory for this step, optionally with appliance name and suffix.
        
        Supports appliance_name for organizing logs by device type and suffix for model/method.
        - Directory name: {step_name}_{suffix} (if suffix exists) or {step_name}
        
        Args:
            context (dict): Shared context containing sequence_id and log_root
            
        Returns:
            str: Path to step's log directory
        """
        log_root = context.get('log_root', os.path.join('log', 'default'))
        
        step_folder_name = self.name
        if self.suffix:
            step_folder_name = f"{self.name}_{self.suffix}"
            
        step_log_dir = os.path.join(log_root, step_folder_name)
        os.makedirs(step_log_dir, exist_ok=True)
        return step_log_dir
