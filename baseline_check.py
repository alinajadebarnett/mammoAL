"""
experiment.py Version 3
Start Date: 7/08/23
"""

# Python Library imports
import argparse
from datetime import datetime
from file_organization import update_dir_with_oracle_info_initial
import glob
import numpy as np
import os
import pandas as pd
import random
import SimpleITK as sitk
import shutil
import sys
import torch
from tqdm import tqdm
import cv2
import pickle

# Backend py file imports
from file_organization import save_files_for_nnunet
from file_organization import convert_initial_segmentations_to_numpy, change_image_paths_to_follow_file_structure
from dataloader import get_DataLoader
from disc_model import disc_model
from auto_oracle import query_oracle_automatic_no_threshold
from manual_oracle import save_oracle_results
import seg_model
import unet_model
import nnunet_model
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss

# Discontinued imports
# from file_organization import save_active_learning_results, update_dir_with_oracle_info, save_active_learning_results_all
# from file_organization import redirect_saved_oracle_filepaths_to_thresheld_directory, remove_bad_oracle_results
# from auto_oracle import query_oracle_automatic

def active_learning_experiment(run_id, oracle_query_method, task_id):
    # EXPERIMENT SETUP
    iter_num = 0
    run_dir = os.path.join(OUTPUT_DIR, f"Run_{run_id}")
    run_loop_dir, save_dir, save_npy_dir, init_save_dir, correct_save_dir, valid_output_dir, discriminator_input_dir = get_iter_directories(run_dir, iter_num)
    valid_input_dir = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task103_train", 'imagesTs')

    # PATH SETUP
    cbis_ddsm_dir =  "/usr/xtmp/gk122/mammoproj/data/cbis_ddsm_data/stacks_npy"                                         # Path to CBIS-DDSM data    (2, 224, 224)
    # in_house_dir = "/usr/xtmp/gk122/mammoproj/data/in_house/images_npy"                                               # Path to in-house data     (1, 640, 640)
    in_house_dir = "/usr/xtmp/vs196/nn_data/nnUNet_raw_data_base/nnUNet_raw_data/Task103_train/imagesTr"
    # ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset_corrected/train/"                                            # Path to ground-truth data (2, 224, 224)
    ground_truth_dir =  "/usr/xtmp/vs196/mammoproj/final_dataset_corrected/train/"


    # Initialize discriminator
    print(f"RUN_DIR: {run_dir}")
    print("\n===INITIALIZING DISCRIMINATOR WITH CBIS-DDSM===")
    # discriminator = get_init_discriminator(training_dir=cbis_ddsm_dir,
    #                                        batch_size=BATCH_SIZE_INIT,
    #                                        num_epochs=INITIAL_DISCRIMINATOR_EPOCHS)
    
    # Load initial segmenter and get initial validation accuracy
    print("\n\n\n\n===LOADING INITIAL SEGMENTER (PRETRAINED ON CBIS-DDSM)===")
    model_save_path = os.path.join(run_dir, 'all', "Iter"+str(iter_num)+".model")

    segmenter = nnunet_model.nnunet_model(base_model_task_id = "Task901_cbis-ddsm")
    # TODO: Get baseline model without needing to pass in dataset
    segmenter.load_model("/usr/xtmp/vs196/nn_data/nnUNet_preprocessed/Task901_cbis-ddsm") # Get base model
    print(f"Epoch before resetting: {segmenter.trainer.epoch}.", flush=True)
    #segmenter.trainer.epoch = 0
    #segmenter.trainer.initial_lr = 1e-3
    print(f"Epoch after resetting: {segmenter.trainer.epoch}.", flush=True)
    segmenter.save_model(model_save_path)

    validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
    print(f"\nMetric of initial segmenter is: {validation_metric}.")

    return

# HELPER FUNCTIONS BELOW
def get_iter_directories(run_dir, iter_num):
    run_loop_dir = os.path.join(run_dir, f"Iter{iter_num}")    
    save_dir = os.path.join(run_loop_dir, "Segmentations")                                                                    # Path to segmentations of iteration in .nii.gz format
    save_npy_dir = os.path.join(run_loop_dir, "Segmentations_npy")                                                            # Path to segmentations of iteration in .npy format
    init_save_dir = os.path.join(run_loop_dir, "InitSegmentations")
    correct_save_dir = os.path.join(run_loop_dir, "Segmentations_C")                                                             # Top file of AL iteration
    valid_output_dir = os.path.join(run_loop_dir, "ValSegmentations")
    discriminator_input_dir = os.path.join(run_loop_dir, "Segmentations_npy_stacks")

    return run_loop_dir, save_dir, save_npy_dir, init_save_dir, correct_save_dir, valid_output_dir, discriminator_input_dir

# Saves initial segmentations into save_dir
def save_init_segmentations(save_dir):
    prev_dir = "/usr/xtmp/sc834/mammoproj/nnunet_integration_tmp/AllOracleRuns/Run_7_18_get_initial_segmentations_30/random/Iter_1_20/initial_segmentations"
    
    # Copy the existing predictions into working folder - TODO: Do not use rmdir
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)
    
    shutil.copytree(prev_dir, save_dir)

# Saves new segmentations into save_npy_dir
def convert_segmentations(save_dir, save_npy_dir, ground_truth_dir):
    # Deletes and creates directory to save .npy files
    if os.path.exists(save_npy_dir):
        shutil.rmtree(save_npy_dir, ignore_errors=True)

    # Convert initial segmentations into numpy format
    convert_initial_segmentations_to_numpy(save_dir, save_npy_dir)

    # Restructure Segmentations_npy to follow Irregular, Round, Oval structure
    # NOTE: This solution should only be temporary - in real life, cannot structure output images into categories without the oracle.
    change_image_paths_to_follow_file_structure(save_npy_dir, ground_truth_dir)

# Returns discrmimator trained on training_dir
def get_init_discriminator(training_dir, batch_size, num_epochs):
    dataloader = get_DataLoader(training_dir, batch_size, 2)

    discriminator = disc_model()
    discriminator.load_model(dataloader)
    discriminator.initialize_model_minimize_iou(batch_size=batch_size, epochs=num_epochs) # initial training

    return discriminator

# Get discriminator to produce scores on segmentations, from 0 to 1 (bad to good segmentation)
def get_patient_scores(discriminator, input_dir, batch_size):
    segmentation_dataloader = get_DataLoader(input_dir, batch_size, 2)
    patient_scores = discriminator.get_scores(segmentation_dataloader) # sorted
    
    return patient_scores

def validate(run_dir, segmenter):
    print("=== CREATING SEGMENTATIONS FOR TEST SET ===")
    if UNET:
        valid_input_dir =  f"/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/"
    else:
        valid_input_dir = os.path.join(
            os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task505_duke_mammo_corrected_0821", 'imagesTs')
    
    valid_output_dir = os.path.join(run_dir, "ValSegmentations")
    validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
    print(f"Metric of new segmenter after active learning is: {validation_metric}.")
   
    return validation_metric

def save_oracle_results_no_threshold(oracle_results, save_dir, segmentation_dir, ground_truth_dir):
    good_save_paths = []
    bad_save_paths = []
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for patient in oracle_results.keys():
        #patient is the patient id
        #oracle_results[patient] is 0/1
        if(oracle_results[patient]==1):
            #good segmentation - save into another folder
            if(patient.startswith("/")):
                save_path = os.path.join(save_dir, patient[1:] + ".npy")
            else:
                save_path = os.path.join(save_dir, patient + ".npy")
            load_path = os.path.join(segmentation_dir, patient + ".npy")
            shape_type = load_path.split("/")[-2]
            im = np.load(load_path)
            save_dir_dir = os.path.join(save_dir, shape_type)
            if not os.path.exists(save_dir_dir):
                os.makedirs(save_dir_dir)
            np.save(save_path,im)
            good_save_paths.append(save_path)
        else:
            #bad segmentation - get ground truth mask
            if(patient.startswith("/")):
                save_path = os.path.join(save_dir, patient[1:] + ".npy")
            else:
                save_path = os.path.join(save_dir, patient + ".npy")
            load_path = os.path.join(ground_truth_dir, patient + ".npy")
            shape_type = load_path.split("/")[-2]
            im = np.load(load_path)
            save_dir_dir = os.path.join(save_dir, shape_type)
            if not os.path.exists(save_dir_dir):
                os.makedirs(save_dir_dir)
            np.save(save_path,im)
            bad_save_paths.append(save_path)
    print("Done with saving this iteration of oracle results")
    return good_save_paths, bad_save_paths

def run_experiment(oracle_query_method, task_id):
    start = datetime.now()
    print("Starting run, and timing")

    # best, worst, random, uniform
    # oracle_query_method = "best"

    run_unique_id = f"{RUN_ID}_{RANDOM_SEED_NUMBER}"

    # Print experiment information at the beginning
    print("===INFORMATION ABOUT THE EXPERIMENT=== \n")
    exp_config_text = f"""Experiment date/time EST: {datetime.now()}
    Provided run_id: {run_unique_id} \n
    Random seed: {RANDOM_SEED_NUMBER} \n
    AL Query Methods: {oracle_query_method} \n
    Initial batch size for discriminator: {BATCH_SIZE_INIT} \n
    Initial discriminator training # epochs: {INITIAL_DISCRIMINATOR_EPOCHS} \n
    Updating discriminator # epochs: {UPDATE_DISCRIMINATOR_EPOCHS} \n
    Update segmenter # epochs: {SEGMENTER_UPDATE_EPOCHS_INIT} \n
    STARTING TASK ID: {task_id} \n"""
    
    print(exp_config_text)
    
    save_dir = os.path.join(OUTPUT_DIR, run_unique_id)
    # Save experiment configuration as a text file
    if os.path.exists(save_dir):
        result = input("The output directory already exists, delete it and proceed? (y/n)")
        if result == 'y':
            shutil.rmtree(save_dir)
            print("Done")
            os.mkdir(save_dir)
            with open(os.path.join(save_dir, "exp_configuration.txt"), "w") as file:
                file.write(exp_config_text)
        else:
            raise Exception("Use a different run id.")
    else:
        os.mkdir(save_dir)
        with open(os.path.join(save_dir, "exp_configuration.txt"), "w") as file:
            file.write(exp_config_text)

    # Begin the experiment
    active_learning_experiment(run_id=run_unique_id,
                                oracle_query_method=oracle_query_method,
                                task_id=task_id)

    print("Finished run")
    print(datetime.now()-start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', "--run_id", required = True)
    parser.add_argument('-o', "--output_dir", required = True)
    parser.add_argument("--random_seed", nargs=1, type=int, required = True)
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--nnunet', dest='unet', action='store_false')
    parser.add_argument('--task_id', nargs=1, type=int)
    parser.add_argument('-q', '--query_method', nargs=1)
    parser.set_defaults(task_id=[-1])
    parser.set_defaults(unet=False)
    parser.set_defaults(query_method="best")
    args = parser.parse_args()

    query_method = args.query_method[0]

    task_id = args.task_id[0]
    if task_id == -1:
        print("Default Task ID")
        last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')), key=lambda x: int((x.split('/')[-1]).split('_')[0][4:]))[-1]
        last_task_filename = last_task.split('/')[-1]
        last_task = last_task_filename.split('_')[0][4:]
        task_id = int(last_task) + 1
    print(f"Task ID: {task_id}")

    RANDOM_SEED_NUMBER = args.random_seed[0]
    torch.manual_seed(RANDOM_SEED_NUMBER)
    torch.cuda.manual_seed(RANDOM_SEED_NUMBER)
    np.random.seed(RANDOM_SEED_NUMBER)
    random.seed(RANDOM_SEED_NUMBER)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    # Originally just args.run_id - added random_seed to allow testing across different seeds
    RUN_ID = args.run_id
    OUTPUT_DIR = args.output_dir
    UNET = args.unet

    # TUNABLE PARAMETER CONSTANTS
    BATCH_SIZE_INIT = 32
    INITIAL_DISCRIMINATOR_EPOCHS = 10
    UPDATE_DISCRIMINATOR_EPOCHS = 5
    SEGMENTER_UPDATE_EPOCHS_INIT = 5
    IOU_THRESHOLD = 0.7

    # RUN EXPERIMENT
    run_experiment(query_method, task_id)