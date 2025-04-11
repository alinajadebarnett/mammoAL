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

# Discontinued imports
# from file_organization import save_active_learning_results, update_dir_with_oracle_info, save_active_learning_results_all
# from file_organization import redirect_saved_oracle_filepaths_to_thresheld_directory, remove_bad_oracle_results
# from auto_oracle import query_oracle_automatic

def active_learning_experiment(run_id, oracle_query_method, task_id):
    # EXPERIMENT SETUP
    iter_num = 0
    run_dir = os.path.join(OUTPUT_DIR, f"Run_{run_id}")
    run_loop_dir, save_dir, save_npy_dir, init_save_dir, correct_save_dir, valid_output_dir, discriminator_input_dir = get_iter_directories(run_dir, iter_num)
    valid_input_dir = os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task505_duke_mammo_corrected_0821", 'imagesTs')

    # PATH SETUP
    cbis_ddsm_dir = "/usr/xtmp/sc834/mammoproj/data/cbis_ddsm_data/stacks_npy"                                          # Path to CBIS-DDSM data    (2, 224, 224)
    # in_house_dir = "/usr/xtmp/gk122/mammoproj/data/in_house/images_npy"                                               # Path to in-house data     (1, 640, 640)
    in_house_dir = "/usr/xtmp/sc834/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task505_duke_mammo_corrected_0821/imagesTr"
    # ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset_corrected/train/"                                            # Path to ground-truth data (2, 224, 224)
    ground_truth_dir = "/usr/xtmp/gk122/mammoproj/data/verified_dataset/train/"


    # Initialize discriminator
    print(f"RUN_DIR: {run_dir}")
    print("\n===INITIALIZING DISCRIMINATOR WITH CBIS-DDSM===")
    discriminator = get_init_discriminator(training_dir=cbis_ddsm_dir,
                                           batch_size=BATCH_SIZE_INIT,
                                           num_epochs=INITIAL_DISCRIMINATOR_EPOCHS)
    
    # Load initial segmenter and get initial validation accuracy
    print("\n\n\n\n===LOADING INITIAL SEGMENTER (PRETRAINED ON CBIS-DDSM)===")
    model_save_path = os.path.join(run_dir, 'all', "Iter"+str(iter_num)+".model")

    segmenter = nnunet_model.nnunet_model()
    # TODO: Get baseline model without needing to pass in dataset
    segmenter.load_model("/usr/xtmp/sc834/mammoproj/data/nnUNet_preprocessed/Task501_cbis-ddsm") # Get base model
    segmenter.save_model(model_save_path)

    validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
    print(f"\nMetric of initial segmenter is: {validation_metric}.")

    # Get initial labels from initial segmenter, and convert them into image-mask numpy stacks
    print("\n===GET INITIAL LABELS FROM INITIAL SEGMENTER===")
    predict = True
    if predict:
        print("Creating predictions...")
        segmenter.predict(in_house_dir, init_save_dir, correct_save_dir=None, saved_oracle_filepaths={})
    else:
        print("Copying initial labels from another directory...")
        save_init_segmentations(save_dir=init_save_dir)

    # Convert predictions from nii.gz to numpy and stack them with original image to create (2, 244, 244) input shape
    print("\n===CONVERTING INITIAL SEGMENTATION PREDICTIONS INTO NUMPY===")
    convert_segmentations(save_dir=init_save_dir,
                          save_npy_dir=save_npy_dir,
                          ground_truth_dir=ground_truth_dir)
    
    print("\n===UPDATE DISCRIMINATOR INPUT DIRECTORY WITH INITIAL SEGMENTATIONS===")
    # Saves npy stack of original image and predicted mask, (2, 224, 224) to save_dir
    update_dir_with_oracle_info_initial(save_dir=discriminator_input_dir, im_dir=ground_truth_dir, mask_dir=save_npy_dir)

    # pandas dataframe where columns are query_type query_number IOU location of saved model
    segmenter_experiment_output = pd.DataFrame(columns=['random_seed', 'query_type', 'num_manually_labelled', 'img_seen', 'iter_num', 'IOU', 'saved_model_location'])
    segmenter_experiment_output.to_csv(os.path.join(run_dir, 'segmenter_output.csv'), sep=',')

    manually_labelled_experiment_output = pd.DataFrame(columns=['random_seed', 'query_type', 'num_manually_labelled', 'img_seen'])
    manually_labelled_experiment_output.to_csv(os.path.join(run_dir, 'manually_labelled_output.csv'), sep=',')

    # BEGINNING OF ACTIVE LEARING ITERATION STAGE
    print("\n\n-----------")
    print("\n===BEGINNING OF THE ACTIVE LEARNING LOOP ===\n")
    print("-----------")
    oracle_results = {}
    prev_oracle_results = {}
    new_oracle_results = {}
    oracle_with_ground_truth_results = {}
    saved_oracle_filepaths = []
    total_img_seen = 0
    query_number = 20
    num_manually_labelled = 0
    # num_needed_correct = 11
    all_patient_scores = []

    # Segmenter update loop
    while True:
        good_saved_oracle_filepaths = []
        bad_saved_oracle_filepaths = []

        # Discrimnator update loop
        while True:
            # If initial iteration already initiated
            if total_img_seen > 0:
                print(f"===UPDATING DISCRIMINATOR (total images: {total_img_seen})===")
                # Update discriminator using incremental original masks and ground truth masks
                discriminator_update_dir = os.path.join(run_dir, "OracleSegmentations_npy_stacks")

                if not os.path.exists(discriminator_update_dir):
                    os.makedirs(discriminator_update_dir)
                    os.makedirs(discriminator_update_dir + "/Irregular")
                    os.makedirs(discriminator_update_dir + "/Oval")
                    os.makedirs(discriminator_update_dir + "/Round")

                pat_ids = list(new_oracle_results.keys())
                for pat_id in tqdm(pat_ids):
                    oracle_with_ground_truth_results[pat_id] = new_oracle_results[pat_id]
                    old_path = os.path.join(discriminator_input_dir, pat_id + ".npy")
                    new_path = os.path.join(discriminator_update_dir, pat_id + ".npy")
                    shutil.copyfile(old_path, new_path)

                    if new_oracle_results[pat_id] == 0:
                        new_pat_id = pat_id + "_gt"
                        oracle_with_ground_truth_results[new_pat_id] = 1
                        old_path = os.path.join(ground_truth_dir, pat_id + ".npy")
                        new_path = os.path.join(discriminator_update_dir, new_pat_id + ".npy")
                        shutil.copyfile(old_path, new_path)

                print(f"Using {len(oracle_with_ground_truth_results)} images to retrain discriminator")

                # Updating classifier 1 epoch at a time for 5 epochs.
                for i in range(UPDATE_DISCRIMINATOR_EPOCHS):
                    oracle_results_dataloader = get_DataLoader(discriminator_update_dir, BATCH_SIZE_INIT, 2)
                    discriminator.update_model_dataloader(oracle_results_dataloader, oracle_with_ground_truth_results, batch_size=BATCH_SIZE_INIT, num_epochs=1)

            print("\n\n===GENERATING DISCRIMINATOR SCORES===")
            discriminator_dataloader = get_DataLoader(discriminator_input_dir, BATCH_SIZE_INIT, 2)
            patient_scores = discriminator.get_scores(discriminator_dataloader) # sorted

            all_patient_scores.append(patient_scores)
            print(f"PATIENT SCORES: \n")
            # print(patient_scores)
            print("First 10 patient_scores: " + str(dict(list(patient_scores.items())[0: 10])))
            print("Last 10 patient_scores: " + str(dict(list(patient_scores.items())[-10: ])))
            print("Middle 10 patient_scores: " + str(dict(list(patient_scores.items())[len(patient_scores)//2-5: len(patient_scores)//2+5])))
            print("10 Near Percentile 0.8: " + str(dict(list(patient_scores.items())[int(len(patient_scores)*0.8)-5: int(len(patient_scores)*0.8)+5])))


            print("\n\n===QUERYING THE ORACLE===")
            new_oracle_results = {}
            prev_oracle_results = oracle_results.copy()
            oracle_results = query_oracle_automatic_no_threshold(oracle_results=oracle_results,
                                                                 patient_scores=patient_scores,
                                                                 ground_truth_dir=ground_truth_dir,
                                                                 segmentation_dir=discriminator_input_dir,
                                                                 query_method=oracle_query_method,
                                                                 query_number=query_number,
                                                                 iou_threshold=IOU_THRESHOLD)
            
            total_img_seen = len(oracle_results)
            
            for id in oracle_results.keys():
                if not id in prev_oracle_results.keys():
                    new_oracle_results[id] = oracle_results[id]
            
            print(f"Oracle results for current iter:\n {new_oracle_results}\n")
            print("# new oracled images: ", len(new_oracle_results))
            print("# total oracled images: ", len(oracle_results))

            print("\n\n===SAVING CLASSIFIED ORACLE RESULTS===")
            # Space for saving oracle results and pickling data structures
            new_good_saved_filepaths, new_bad_saved_filepaths = save_oracle_results_no_threshold(oracle_results=new_oracle_results,
                                                                                                 save_dir=correct_save_dir,
                                                                                                 segmentation_dir=discriminator_input_dir,
                                                                                                 ground_truth_dir=ground_truth_dir)
            good_saved_oracle_filepaths += new_good_saved_filepaths    # Segmenter masks
            bad_saved_oracle_filepaths += new_bad_saved_filepaths      # Ground truth masks
            num_manually_labelled += len(new_bad_saved_filepaths)

            manually_labelled_experiment_output = pd.DataFrame([{'random_seed': RANDOM_SEED_NUMBER,
                                                                 'query_type': oracle_query_method,
                                                                 'num_manually_labelled': num_manually_labelled,
                                                                 'img_seen': total_img_seen,}])
            manually_labelled_experiment_output.to_csv(os.path.join(run_dir, 'manually_labelled_output.csv'), sep=',', mode='a', header=False)

            # If all images seen, exit loop
            if len(new_oracle_results) != query_number:
                print("No more images to oracle")
                print(f"\nExiting loop {iter_num}")
                break
            # # If not enough images are classified as correct by oracle, exit loop (convergence criteria)
            # elif len(new_good_saved_filepaths) < num_needed_correct:
            #     print(f"Oracle classifies {len(new_good_saved_filepaths)} images as correct.")
            #     print("Not enough oracle results classified as correct.")
            #     print(f"Needed to classify {num_needed_correct} images as correct.")
            #     print(f"\nExiting loop {iter_num}")
            #     break
            else:
                print(f"Oracle classifies {len(new_good_saved_filepaths)} images as correct.\n")
                print(new_good_saved_filepaths)
                break
        
        print("\n=====")
        print(f"Done with loop {iter_num} for query method {oracle_query_method}, random seed {RANDOM_SEED_NUMBER}")
        print(f"# total approved labels: {len(oracle_results)-num_manually_labelled}")
        print(f"# total manually labelled: {num_manually_labelled}")
        print("Moving onto segmenter and discriminator update...")
        print("=====\n")

        segmenter_experiment_output = pd.DataFrame([{'random_seed': RANDOM_SEED_NUMBER,
                                                     'query_type': oracle_query_method,
                                                     'num_manually_labelled': num_manually_labelled,
                                                     'img_seen': total_img_seen,
                                                     'iter_num': iter_num,
                                                     'IOU': validation_metric,
                                                     'saved_model_location': model_save_path}])
        segmenter_experiment_output.to_csv(os.path.join(run_dir, 'segmenter_output.csv'), sep=',', mode='a', header=False)

        with open(os.path.join(run_loop_dir, f"all_patient_scores_dict.pickle"), 'wb') as file:
            pickle.dump(all_patient_scores, file)

        # When all images in the dataset is seen, end active learning iterations
        if len(new_oracle_results) != query_number:
            print("Done querying through all images")
            print(f"Finishing AL experiment for query method {oracle_query_method} on random seed {RANDOM_SEED_NUMBER}")
            break

        # Increase iteration number
        iter_num += 1
        
        run_loop_dir, save_dir, save_npy_dir, init_save_dir, correct_save_dir, valid_output_dir, discriminator_input_dir = get_iter_directories(run_dir, iter_num)

        # Do += to increment over all oracle results, = for only new oracle results
        saved_oracle_filepaths += good_saved_oracle_filepaths + bad_saved_oracle_filepaths

        # SEGMENTATION STAGE
        # Preprocess data with information learned from active learning.
        print("===PREPARING FOR SEGMENTATION STAGE===")
        save_files_for_nnunet(task_id, run_id, saved_oracle_filepaths)

        print("\n\n===UPDATE SEGMENTER WITH ORACLED IMAGES===")
        model_save_path = os.path.join(run_dir, 'all', "Iter" + str(iter_num) +".model")
        segmenter_train_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Task{task_id}_{run_id}')
        segmenter.load_new_data(segmenter_train_dir)
        segmenter.update_model(num_epochs=SEGMENTER_UPDATE_EPOCHS_INIT)
        segmenter.save_model(model_save_path)
        task_id += 1

        # Validate segmenter
        validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
        print(f"Metric of current segmenter is: {validation_metric}.")

        print("\n===PRODUCING SEGMENTATIONS ON TRAIN SET WITH NEW SEGMENTER===")
        segmenter.predict(in_house_dir, init_save_dir, correct_save_dir=correct_save_dir, saved_oracle_filepaths={})

        print("\n===CONVERTING SEGMENTATION PREDICTIONS INTO NUMPY===")
        # Convert predictions from nii.gz to numpy and stack them with original image to create (2, 244, 244) input shape
        convert_segmentations(save_dir=init_save_dir,
                          save_npy_dir=save_npy_dir,
                          ground_truth_dir=ground_truth_dir)

        print("\n===UPDATE DISCRIMINATOR INPUT DIRECTORY WITH NEW SEGMENTATIONS===")
        # Saves npy stack of original image and predicted mask, (2, 224, 224) to save_dir
        update_dir_with_oracle_info_initial(save_dir=discriminator_input_dir, im_dir=ground_truth_dir, mask_dir=save_npy_dir)


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