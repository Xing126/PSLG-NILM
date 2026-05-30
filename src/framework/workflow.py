import datetime
import os
import shutil
import traceback
from typing import List, Type
from .step import Step
from .logger import setup_logger

class Workflow:
    """
    Manages the sequential execution of ML workflow steps.
    """
    def __init__(self, name: str, appliance_name: str = "", sequence_id: str | None = None, resume: bool = False, save_interval: int = 0):
        self.name = name
        self.appliance_name = str(appliance_name).strip()
        self.steps: List[Step] = []
        self.sequence_id = sequence_id or self._generate_sequence_id()
        self.resume = bool(resume)
        self.save_interval = int(save_interval)
        run_id = self._build_run_id()
        self.context = {
            'sequence_id': self.sequence_id,
            'appliance_name': self.appliance_name,
            'run_id': run_id,
            'save_interval': self.save_interval,
            'input_root': 'input',
            'log_root': os.path.join('log', run_id),
            'output_root': os.path.join('output', run_id),
            'data': {} # Stores intermediate results
        }
        
        # Ensure directories exist for the specific run
        self._init_dirs()
        self.logger = setup_logger(self.name, self.context['log_root'])
        self.logger.info(f"Initialized Workflow: {self.name} with ID: {self.sequence_id}")

    def _generate_sequence_id(self) -> str:
        """Generates a unique ID based on the current timestamp."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _build_run_id(self) -> str:
        """Builds run id as appliance_timestamp when appliance name is provided."""
        if self.appliance_name:
            return f"{self.appliance_name}_{self.sequence_id}"
        return self.sequence_id

    def _init_dirs(self):
        """Creates output subdirectories according to specification."""
        # Log directory (cache)
        os.makedirs(self.context['log_root'], exist_ok=True)
        
        # Output subdirectories
        os.makedirs(os.path.join(self.context['output_root'], 'output'), exist_ok=True)
        os.makedirs(os.path.join(self.context['output_root'], 'figure'), exist_ok=True)

    def add_step(self, step: Step):
        """Adds a step to the workflow."""
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")

    def run(self):
        """Runs all steps in the order they were added."""
        self.logger.info("Starting workflow execution...")
        force_run = False
        try:
            for i, step in enumerate(self.steps, 1):
                self.logger.info(f"--- Step {i}: {step.name} ---")
                
                # Update context with current step log dir
                step_log_dir = step.get_log_dir(self.context)
                done_flag_path = os.path.join(step_log_dir, ".done")
                
                # Resume logic: skip if done AND not forced by previous step change
                if self.resume and not force_run and os.path.exists(done_flag_path):
                    self.context = step.restore(self.context)
                    self.logger.info(f"Step {step.name} skipped (already done).")
                    continue
                
                # Execute step
                self.context = step.run(self.context)
                with open(done_flag_path, "w", encoding="utf-8") as f:
                    f.write(self.sequence_id)
                
                # If a step is actually executed, force all subsequent steps to run to ensure consistency
                if self.resume:
                    force_run = True
                    self.logger.info(f"Step {step.name} executed, will force subsequent steps to run for consistency.")
                
                self.logger.info(f"Step {step.name} completed successfully.")
            
            self.logger.info("Workflow execution finished successfully.")
            
        except Exception as e:
            self.logger.error(f"Workflow failed at step {step.name if 'step' in locals() else 'initialization'}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.info(f"Execution logs and artifacts stored in {self.context['log_root']} and {self.context['output_root']}")
