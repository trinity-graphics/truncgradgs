#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import PIL.Image
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def migrate_tb_to_wandb(tb_log_dir, wandb_project, wandb_run_name=None, wandb_tags=None):
    """
    Migrate TensorBoard logs to Weights & Biases.
    """
    if wandb_run_name is None:
        wandb_run_name = os.path.basename(tb_log_dir)
    
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        tags=wandb_tags or ["migrated_from_tb"],
        config={"migrated_from": tb_log_dir},
        resume="never",
    )
    
    print(f"Loading TensorBoard events from {tb_log_dir}")
    ea = EventAccumulator(
        tb_log_dir,
        size_guidance={'scalars': 0, 'histograms': 0, 'images': 0}
    )
    ea.Reload()
    
    # Collect all events
    all_events = []
    
    # Scalars
    scalar_tags = ea.Tags()['scalars']
    print(f"Found {len(scalar_tags)} scalar tags")
    for tag in scalar_tags:
        for event in ea.Scalars(tag):
            all_events.append(('scalar', event.step, tag, event.value))
    
    # Histograms
    hist_tags = ea.Tags()['histograms']
    if hist_tags:
        print(f"Found {len(hist_tags)} histogram tags")
        for tag in hist_tags:
            for hist in ea.Histograms(tag):
                all_events.append(('histogram', hist.step, tag, hist))
    
    # Images
    image_tags = ea.Tags()['images']
    if image_tags:
        print(f"Found {len(image_tags)} image tags")
        for tag in image_tags:
            for img in ea.Images(tag):
                all_events.append(('image', img.step, tag, img))
    
    # Sort by step
    all_events.sort(key=lambda x: x[1])
    print(f"Total events: {len(all_events)}")
    
    # Log grouped by step
    current_step = None
    step_data = {}
    
    for event_type, step, tag, data in all_events:
        if step != current_step:
            if current_step is not None and step_data:
                wandb_run.log(step_data, step=current_step)
            current_step = step
            step_data = {}
        
        if event_type == 'scalar':
            step_data[tag] = data
        elif event_type == 'histogram':
            bucket_limits = list(data.histogram_value.bucket_limit)
            bucket_counts = list(data.histogram_value.bucket)
            if len(bucket_limits) == len(bucket_counts):
                bucket_limits.append(bucket_limits[-1] + 1e-6)
            step_data[tag] = wandb.Histogram(np_histogram=(bucket_counts, bucket_limits))
        elif event_type == 'image':
            img_pil = PIL.Image.open(BytesIO(data.encoded_image_string))
            step_data[tag] = wandb.Image(img_pil)
    
    # Log last step
    if current_step is not None and step_data:
        wandb_run.log(step_data, step=current_step)
    
    print(f"Migration complete! View at: {wandb_run.url}")
    wandb.finish()
    print("Wandb run finished - no more updates will be accepted")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate TensorBoard logs to Weights & Biases")
    parser.add_argument("--tb_log_dir", type=str, required=True,
                        help="Path to the directory containing TensorBoard event files")
    parser.add_argument("--wandb_project", type=str, default="3DGS-frame-to-frame-XP",
                        help="Name of the wandb project")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Name for the wandb run (defaults to directory name)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None,
                        help="Tags for the wandb run")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.tb_log_dir):
        print(f"Error: {args.tb_log_dir} is not a valid directory")
        sys.exit(1)
    
    migrate_tb_to_wandb(
        tb_log_dir=args.tb_log_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )
