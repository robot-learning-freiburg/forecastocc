import os
import time

import torch.distributed as dist
from mmcv.runner import HOOKS, Hook

try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class WandbCustomLoggerHook(Hook):
    """Custom MMDet3D WandB Logger Hook"""

    def __init__(self, interval=10, project="mmdet3d_project", name="experiment_1", task='occupancy', log_auxiliary_loss=False, log_checkpoint=False, log_checkpoint_metadata=False, log_samples=-1):
        self.interval = interval
        self.project = project
        self.name = name
        self.initialized = False
        self.task = task  
        self.log_auxiliary_loss = log_auxiliary_loss 
        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = log_checkpoint_metadata
        self.output_buffer = {}  # Buffer to store outputs for logging
        self.log_samples = log_samples  # Number of samples to log, -1 means None

    def is_master_process(self):
        """Check if this process is the main (rank 0) process in distributed training."""
        return not dist.is_initialized() or dist.get_rank() == 0  # True for single-GPU

    def before_run(self, runner):
        """Initialize WandB only on the master process."""
        if self.initialized:
            return  # Prevent double initialization

        if self.is_master_process():  # Only initialize on rank 0
            if wandb is None:
                raise ImportError(
                    'WandbCustomLoggerHook requires the optional "wandb" '
                    'dependency.')
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            experiment_name = f"{self.name}_{timestamp}"

            self.work_dir = runner.work_dir  # Use MMDet3D's workdir
            os.makedirs(self.work_dir, exist_ok=True)

            # Initialize WandB only once
            wandb.init(
                project=self.project,
                name=experiment_name,
                dir=self.work_dir
            )

            runner.logger.info(f"[WandB] Initialized on rank 0: {experiment_name}")
            runner.logger.info(f"[WandB] Initialized for task: {self.task}")

        # Ensure all processes wait for rank 0 to finish initialization
        if dist.is_initialized():
            dist.barrier()  # Sync all processes

        self.initialized = True  # Prevent duplicate init

    def after_train_iter(self, runner):
        """Log training metrics every `interval` iterations."""
        if self.is_master_process():
            if runner.iter % self.interval == 0:
                log_data = {
                    "loss": runner.outputs["loss"].item(),
                    "learning_rate": runner.current_lr()[0],
                    "iteration": runner.iter,
                    "epoch": runner.epoch
                }
                # Include all available loss values from log_vars
                if self.log_auxiliary_loss:
                    for key, value in runner.outputs["log_vars"].items():
                        log_data[key] = value
                else:
                    for key, value in runner.outputs["log_vars"].items():
                        if key in ['loss_depth', 'loss_mask', 'loss_cls', 'loss_dice', 'loss_consistency' ]:
                            log_data[key] = value
                    
                wandb.log(log_data)

    def after_train_epoch(self, runner):
        """Log at the end of each epoch."""
        if self.is_master_process():  # Ensure only rank 0 logs
            log_data = {
                    "loss": runner.outputs["loss"].item(),
                    "learning_rate": runner.current_lr()[0],
                    "iteration": runner.iter,
                    "epoch": runner.epoch
                }
            # Include all available loss values from log_vars
            for key, value in runner.outputs["log_vars"].items():
                if key in ['loss_depth', 'loss_mask', 'loss_cls', 'loss_dice', 'loss_consistency' ]:
                    log_data[key] = value
            if self.task == 'occupancy':      
                for key, value in self.output_buffer.items():
                    if ("iou" in key or "miou" in key) and "ssc" not in key:
                        log_data[key] = value
                
                output_data = self.output_buffer
                                
                all_iou_ssc_keys = sorted([key for key in output_data.keys() if "iou_ssc" in key])                
                class_names_keys = sorted([key for key in output_data.keys() if "class_names" in key])
                table_data = []
                
                for index in range(len(all_iou_ssc_keys)):
                    if index == 0:
                        class_names = output_data.get(class_names_keys[index], {})
                        iou_ssc = output_data.get(all_iou_ssc_keys[index], {})
                        table_data = [[c_i, iou_ssc_i] for c_i, iou_ssc_i in zip(class_names, iou_ssc)]
                        columns = [class_names_keys[index], all_iou_ssc_keys[index]]
                    else:
                        iou_ssc = output_data.get(all_iou_ssc_keys[index], {})
                        table_data = [t_i + [iou_ssc_i] for t_i, iou_ssc_i in zip(table_data, iou_ssc)]
                        columns.append(all_iou_ssc_keys[index])
                                   
                if table_data:  # Ensure there's data before logging
                    # Determine max width for class name column
                    max_class_length = max(len(str(row[0])) for row in table_data + [["Class"]])
                    
                    # Determine column widths dynamically for each IoU column
                    column_widths = []
                    for col_index in range(1, len(table_data[0])):
                        col_name = columns[col_index]
                        max_width = max(len(col_name), *(len(f"{row[col_index]:.2f}") for row in table_data))
                        column_widths.append(max_width)

                    # Create dynamic markdown header
                    table_md = f"| {'Class'.ljust(max_class_length)} "
                    for idx, col_name in enumerate(columns[1:]):
                        table_md += f"| {col_name.center(column_widths[idx])} "
                    table_md += "|\n"

                    # Construct separator line with proper dash lengths
                    table_md += f"|{'-' * max_class_length}"
                    for width in column_widths:
                        table_md += f"|{'-' * (width + 2)}"  # +2 for padding inside each cell
                    table_md += "|\n"
                    
                    # Fill table rows
                    for row in table_data:
                        row_md = f"| {str(row[0]).ljust(max_class_length)} "
                        for idx, val in enumerate(row[1:]):
                            formatted_val = f"{val:.2f}".center(column_widths[idx])
                            row_md += f"| {formatted_val} "
                        row_md += "|\n"
                        table_md += row_md

                    current_step = wandb.run.step
                    wandb.log({"Validation SSC IoUs (Markdown)": wandb.Html(f"<pre>{table_md}</pre>")}, step=current_step)  
                    
                    if self.log_samples > 0:
                        all_vis_keys = sorted([key for key in output_data.keys() if "vis" in key])
                        for key in all_vis_keys:
                            vis_data = output_data.get(key, [])
                            if vis_data:
                                vis_images = [wandb.Image(img) for img in vis_data]
                                wandb.log({f"Validation {key}": vis_images}, step=current_step)                            
            else:        
                for key, value in self.output_buffer.items():
                    if "IoU" in key or "mIoU" in key:
                        log_data[key] = value
                
                output_data = self.output_buffer
                class_names = {key.split('_')[-2]: value for key, value in output_data.get("SSC_classwise", {}).items()}

                table_data = []
                future_class_names = output_data.get("Future_SSC_classwise", {})

                if future_class_names:
                    for name, ssc_iou in class_names.items():
                        future_ssc_iou = future_class_names.get(f"Future_SSC_{name}_IoU", None)
                        table_data.append([name, ssc_iou, future_ssc_iou])

                    columns = ["Class", "SSC IoU (%)", "Future SSC IoU (%)"]
                else:
                    for name, ssc_iou in class_names.items():
                        table_data.append([name, ssc_iou])

                    columns = ["Class", "SSC IoU (%)"]
                    
                if table_data:  # Ensure there's data before logging
                    max_class_length = max(len(row[0]) for row in table_data + [["Class"]])  # Account for the header
                    column_width = max(len("SSC IoU (%)"), len("Future SSC IoU (%)"))  # Ensure numeric columns are consistent

                    # Create table header with dynamic spacing
                    table_md = f"| {'Class'.ljust(max_class_length)} | {'SSC IoU (%)'.center(column_width)} | {'Future SSC IoU (%)'.center(column_width)} |\n"
                    table_md += f"|{'-' * max_class_length}|{'-' * column_width}|{'-' * column_width}|\n"

                    # Fill table rows with centered values
                    for row in table_data:
                        class_name = row[0].ljust(max_class_length)  # Left-align class names
                        ssc_iou = f"{row[1]:.2f}".center(column_width)  # Center-align numbers
                        future_ssc_iou = f"{row[2]:.2f}".center(column_width) if len(row) > 2 else "-".center(column_width)
                        table_md += f"| {class_name} | {ssc_iou} | {future_ssc_iou} |\n"
                        
                    current_step = wandb.run.step
                    wandb.log({"Validation SSC IoUs (Markdown)": wandb.Html(f"<pre>{table_md}</pre>")}, step=current_step)
                    
            wandb.log(log_data)

    def after_run(self, runner):
        """Close wandb when training finishes."""
        if self.is_master_process(): 
            wandb.finish()
