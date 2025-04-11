"""
Further remodeling to finish the full AL loop. This involves:
- Implementing image stacking, so that a line on the graph can be made from one continuous run of the AL loop, 
not separate runs for each # images seen
- Updating the discriminator with any images queried in a given AL loop, updating AL loop with oracle results
- Restructuring all outputs so that one run of the script produces folders for each query method, each AL loop, etc
- Changing training dataset for initializing discriminator from ground-truth to cbis-ddsm image-mask pairs
"""

# Python Library imports
import argparse
from datetime import datetime
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
import pickle

# Backend py file imports
from file_organization import save_active_learning_results, update_dir_with_oracle_info
from file_organization import redirect_saved_oracle_filepaths_to_thresheld_directory, remove_bad_oracle_results
from file_organization import save_files_for_nnunet
from file_organization import update_dir_with_oracle_info_initial
from file_organization import convert_initial_segmentations_to_numpy, change_image_paths_to_follow_file_structure
from dataloader import get_DataLoader
from disc_model import disc_model
from auto_oracle import query_oracle_automatic, query_oracle_automatic_no_threshold
import seg_model
import unet_model
import nnunet_model

# NOTE: The current implementation only allows nnUNet implementation. Small adjustments can be made to allow UNet.
def active_learning_experiment(imgs_seen_list, run_id, output_dir, oracle_query_method, random_seed, given_task_id=-1, unet=False):

    # FILE DEFINITIONS and SETUP
    # files should be .npy, (2, , ) channels: image, binarized segmentation 
    # Discriminator initialized with cbis-ddsm image/mask pairs, initially of shape (2, 224, 224)
    # discriminator_training_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    discriminator_training_dir = "/usr/xtmp/sc834/mammoproj/data/cbis_ddsm_data/stacks_npy"
    # oracle_query_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset_corrected/train/"
    # ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset_corrected/train/"
    oracle_query_dir = "/usr/xtmp/sc834/mammoproj/Data/final_dataset_corrected_2/train/"
    ground_truth_dir = "/usr/xtmp/sc834/mammoproj/Data/final_dataset_corrected_2/train/"
    segmentation_folder = '/usr/xtmp/sc834/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task504_duke-mammo/imagesTr'
    # This path contains initial segmentations from the segmenter pretrained on cbis-ddsm (824 predictions, in nii.gz format)
    prev_segmentation_save_dir = "/usr/xtmp/sc834/mammoproj/nnunet_integration_tmp/AllOracleRuns/Run_7_18_get_initial_segmentations_30/random/Iter_1_20/initial_segmentations"
    
    total_images_shown = 0

    # store task_id (if -1, then default task id, if else, custom task id)
    # Custom task id is given when running multiple experiments at once to prevent task id conflicts
    task_id = given_task_id
    
    # Keep track of list of patient_sores for visualization of discriminator outputs
    all_patient_scores = []

    # Initialize dict that contains experiment results
    results_dict = {}

    # Initialize dictionary containing oracle results
    oracle_results = {}

    # Manual oracle thresholding not implemented in this version
    # oracle_result_thresholds = {}

    # To keep track of correctly labelled segmentations by the segmenter / manually labelled
    correct_segmentations_ids = []
    correct_segmentations_filepaths = []
    manually_labelled_ids = []
    
    # run_dir with random seed considered
    run_dir = os.path.join(output_dir, f"Run_{run_id}_{random_seed}")

    print(f"RUN_DIR: {run_dir}")
    print("\n===INITIALIZE DISCRIMINATOR===")

    dataloader = get_DataLoader(discriminator_training_dir, BATCH_SIZE_INIT, NUM_WORKERS)

    discriminator = disc_model()
    discriminator.load_model(dataloader)
    print(discriminator.model)
    discriminator.initialize_model_minimize_iou(batch_size=BATCH_SIZE_INIT, epochs=INITIAL_DISCRIMINATOR_EPOCHS)

    # initialize dataloader for ground_truth images - used later for reference in discriminator update
    # ground_truth_dataloader = get_DataLoader(ground_truth_dir, BATCH_SIZE_INIT, NUM_WORKERS)

    print("\n===LOADING INITIAL SEGMENTER (PRETRAINED ON CBIS-DDSM)===")
    # Where the model is saved - needs to take into account query method, as we need separate segmenter model for each query method
    # model_save_path = os.path.join(run_dir, 'all', "Iter" + str(iter_num)+".model")
    # NOTE: Each time model is saved, currently overrides the old one
    model_save_path = os.path.join(run_dir, 'all', f"Run_{run_id}_{oracle_query_method}_{random_seed}.model")

    if unet:
        segmenter = unet_model.unet_model()
    else:
        segmenter = nnunet_model.nnunet_model()

    # load_model automatically loads the CBIS-DDSM baseline. 
    # load_model not necessary yet (as training images not set), but just to save the baseline model at the path
    segmenter.load_model("/usr/xtmp/sc834/mammoproj/data/nnUNet_preprocessed/Task501_cbis-ddsm")
    segmenter.save_model(model_save_path)


    # Implements the main Active Learning Loop
    print("\n===BEGINNING OF THE ACTIVE LEARNING LOOP ===\n")

    # Initiaze list of saved results for segmenter training - increases in length each iteration
    saved_oracle_filepaths = []

    for index in range(len(imgs_seen_list)):

        # segmenter.load_model(os.path.join(run_dir, 'all'))
        # Number of images that need to be queried in this round of AL loop. 
        additional_imgs = imgs_seen_list[index]
        # If this is not the first loop, then calculate number of additional images trained
        if index > 0:
            additional_imgs = imgs_seen_list[index] - imgs_seen_list[index-1]

        iter_count = index + 1

        # Create a folder for each query method, and create subfolders for each AL loop. 
        run_dir_iter = os.path.join(run_dir, f"{oracle_query_method}", f"Iter_{iter_count}_{imgs_seen_list[index]}")

        # Print some information regarding current AL loop.
        print(f"\nITER NUMBER: {iter_count}\n")
        print(f"Number of additional images: {additional_imgs}")
        print(f"Total number of images seen: {imgs_seen_list[index]}")
        print(f"Run dir for current iteration: {run_dir_iter} \n")
        
        
        segmentation_save_dir = os.path.join(run_dir_iter,"initial_segmentations")

        if os.path.exists(segmentation_save_dir):
            shutil.rmtree(segmentation_save_dir, ignore_errors=True)

        print("\n===PRODUCING INITIAL SEGMENTATIONS ON TRAIN SET WITH CURRENT SEGMENTER===")

        # If the segmenter has gone through at least one loop (thus updated), produce new segmentations. 
        if iter_count == 1:
            # Copy over previous initital segmentations for first iteration
            shutil.copytree(prev_segmentation_save_dir, segmentation_save_dir)
        else:
            # Otherwise create new predictions with the current segmenter. 
            os.makedirs(segmentation_save_dir)   
            start_segmenter = datetime.now()
            segmenter.predict(input_folder=segmentation_folder, output_folder=segmentation_save_dir) 
            print(f"Time taken to produce predictions: {round((datetime.now() - start_segmenter).seconds)} s")

        print("\n===CONVERTING INITIAL SEGMENTATION PREDICTIONS INTO NUMPY===")
        
        # Path where images converted from nii.gz to numpy will be stored
        segmentations_score_dir = os.path.join(run_dir_iter, "initial_segmentations_npy")
        discriminator_input_dir = os.path.join(run_dir_iter, "initial_segmentations_npy_stacks")

        # convert initial segmentations into numpy format
        if os.path.exists(segmentations_score_dir):
            shutil.rmtree(segmentations_score_dir, ignore_errors=True)
        if os.path.exists(discriminator_input_dir):
            shutil.rmtree(discriminator_input_dir, ignore_errors=True)
        
        # Convert nii.gz files to numpy
        convert_initial_segmentations_to_numpy(segmentation_save_dir, segmentations_score_dir)
        # NOTE: temporary solution for this task (in real life, cannot automatically reorganize images based on shape)
        change_image_paths_to_follow_file_structure(segmentations_score_dir, ground_truth_dir)
        # Create image/mask stacks TODO: Change function name to be more accurate
        update_dir_with_oracle_info_initial(save_dir=discriminator_input_dir, im_dir=oracle_query_dir, mask_dir=segmentations_score_dir)


        print("\n===GENERATING INITIAL PATIENT SCORES===")

        # Generate patient scores (good/bad segmentations) on output from segmenter
        segmentation_dataloader = get_DataLoader(discriminator_input_dir, BATCH_SIZE_INIT, NUM_WORKERS)
        patient_scores = discriminator.get_scores(segmentation_dataloader) # sorted

        # For debugging - ideally we want wide range of values, with peaks at 0 and 1
        print("First 10 patient_scores: " + str(dict(list(patient_scores.items())[0: 10])))
        print("Last 10 patient_scores: " + str(dict(list(patient_scores.items())[-10: ])))

        # TODO: Keep track of all patient scores and visualize them to assess discriminator performance
        all_patient_scores.append(patient_scores)

        # QUERYING THE ORACLE
        print("\n\n===QUERYING THE ORACLE===")
        # TODO: Allow different divisors to work to determine train cycles, num images queried
        # assert additional_imgs % 10 == 0
        active_learning_train_cycles = additional_imgs // 10
        if additional_imgs % 10 != 0:
            active_learning_train_cycles += 1
        # Begin loop over number of active learning/
        
        for cycle in tqdm(range(active_learning_train_cycles)):
            
            query_number = 10
            if additional_imgs % 10 != 0 and cycle+1 == active_learning_train_cycles:
                query_number = additional_imgs % 10

            # Querying oracle - currently queries {query_cycles} times.
            try:
                oracle_results = query_oracle_automatic_no_threshold(
                    oracle_results, patient_scores,
                    ground_truth_dir, discriminator_input_dir,
                    query_method=oracle_query_method, query_number=query_number, iou_threshold=IOU_THRESHOLD)

                # print(f"ORACLE RESULTS: {oracle_results}")

            except Exception as e:
                print(str(e))
                print("Something went wrong with the automatic oracle query")
                sys.exit(1)

            total_images_shown += query_number

            print(f"===UPDATING DISCRIMINATOR (total images: {total_images_shown})===")
            # for i in range(5):
            #     # Update based on good/bad label on segmenter predictions
            #     discriminator.update_model_with_oracle_results(oracle_results,batch_size = BATCH_SIZE_INIT, num_epochs=UPDATE_DISCRIMINATOR_EPOCHS)

            # TODO: Python dicts maintain order of insertion, but make it foolproof
            oracle_ids_current_iter = list(oracle_results.keys())[-query_number:]
            oracle_results_iter = dict([(key, oracle_results[key]) for key in oracle_ids_current_iter])
            
            print(f"ORACLE RESULTS FOR THIS ITER: {oracle_results_iter}")

            oracle_update_dir = os.path.join(run_dir_iter, f"oracle_cycle_{cycle+1}")
            update_with_oracle_results_dir = os.path.join(oracle_update_dir, f"segmentations_with_feedback")
            update_with_manually_labeled_dir = os.path.join(oracle_update_dir, f"manually_labeled_segmentations")

            for type in ['Irregular', 'Oval', 'Round']:
                os.makedirs(os.path.join(update_with_oracle_results_dir, type))
                os.makedirs(os.path.join(update_with_manually_labeled_dir, type))
            
            for pat_id in list(oracle_results_iter.keys()):
                src = os.path.join(discriminator_input_dir, pat_id+'.npy')
                # os.makedirs(update_with_oracle_results_dir)
                shutil.copyfile(src, os.path.join(update_with_oracle_results_dir, pat_id+'.npy'))

            print("===UPDATING DISCRIMINATOR WITH ORACLE FEEDBACK===")
            disc_update_dataloader = get_DataLoader(update_with_oracle_results_dir, BATCH_SIZE_INIT, 2)
            discriminator.update_model_exp(disc_update_dataloader, oracle_results_iter, batch_size = BATCH_SIZE_INIT, num_epochs=UPDATE_DISCRIMINATOR_EPOCHS)

            for pat_id in list(oracle_results_iter.keys()):
            # if the oracle result is 0 for a given segmentation, also put manually labelled imgs
                if oracle_results_iter[pat_id] == 0:
                    src = os.path.join(ground_truth_dir, pat_id+'.npy')
                    # os.makedirs(update_with_manually_labeled_dir)
                    shutil.copyfile(src, os.path.join(update_with_manually_labeled_dir, pat_id+'.npy'))

                    oracle_results_iter[pat_id] == 1
            
            print("===UPDATING DISCRIMINATOR WITH MANUALLY ORACLED IMAGES===")
            disc_update_dataloader = get_DataLoader(update_with_manually_labeled_dir, BATCH_SIZE_INIT, 2)
            discriminator.update_model_exp(disc_update_dataloader, oracle_results_iter, batch_size = BATCH_SIZE_INIT, num_epochs=UPDATE_DISCRIMINATOR_EPOCHS)

                # patient_scores = discriminator.get_scores(segmentation_dataloader)
                # all_patient_scores.append(patient_scores)

        
        # Slice results that we want to use for this loop - # additional images added at the end of oracle_results
        oracle_results_desired_ids_list = list(oracle_results.keys())[-additional_imgs:]
        
        # For debugging
        # print(f"LENGTH OF ORACLE_RESULTS: {len(oracle_results)}")

        # IN-BETWEEN STAGE
        print("\n===SAVING CLASSIFIED ORACLE RESULTS===")

        # NOTE: Declared here if need to reset saved paths every iteration
        # saved_oracle_filepaths = []

        for pat_id in oracle_results_desired_ids_list:
            # Keep count of correct segmentations by nnunet - 'yes' by the oracle
            # Keep count of manually labelled images - 'no' by the oracle
            if oracle_results[pat_id] == 1:
                filename = os.path.join(discriminator_input_dir, pat_id + '.npy')
                correct_segmentations_ids.append(pat_id)
                correct_segmentations_filepaths.append(filename)
            elif oracle_results[pat_id] == 0:
                filename = os.path.join(ground_truth_dir, pat_id + '.npy')
                manually_labelled_ids.append(pat_id)
            
            saved_oracle_filepaths.append(filename)

        # TODO: Update discriminator also with manually labelled images as data for 1 (or half mismatch)
        # print("\n===FURTHER UPDATE DISCRIMINATOR WITH GOOD LABELS===")
        # print(manually_labelled_ids[-additional_imgs:])
        # discriminator.update_model_with_good_labels(manually_labelled_ids[-additional_imgs:], ground_truth_dataloader, batch_size = BATCH_SIZE_INIT, num_epochs=UPDATE_DISCRIMINATOR_EPOCHS)

        # Update the training file save path to incorporate the AL loop count
        run_id_with_iter_count = str(run_id) + f"_{oracle_query_method}_{iter_count}_{random_seed}"

        # # When task_id is NOT manually assigned
        if not unet: 
            if given_task_id == -1:
                print("Using Default Task ID")
                last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')), key=lambda x: int((x.split('/')[-1]).split('_')[0][4:]))[-1]
                last_task_filename = last_task.split('/')[-1]
                last_task = last_task_filename.split('_')[0][4:]
                task_id = int(last_task) + 1
            else:
                task_id += 1

        save_files_for_nnunet(task_id, run_id_with_iter_count, saved_oracle_filepaths)

        # SEGMENTATION STAGE
        print("\n\n===UPDATING SEGMENTER FOR 5 EPOCHS===")
        # Train model using learned oracle data for 5 epochs
        # learned oracle data = images that are in the "new saved oracle filepaths" (the images that the oracle said looked good)

        # Time the update
        start_update = datetime.now()

        if unet:
            segmenter_train_dir = saved_oracle_filepaths
        else:
            segmenter_train_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Task{task_id}_{run_id_with_iter_count}')
        
        # If it is the first loop, load the initial cbis-ddsm baseline segmenter 
        # Else, update on the existing segmenter (nnUNetTrainerV2.py initialize() method changed)
        if index == 0:
            segmenter.load_model(segmenter_train_dir)
        else:
            segmenter.load_new_data(segmenter_train_dir)
        segmenter.update_model(num_epochs=SEGMENTER_UPDATE_EPOCHS_INIT)

        # Save the updated model
        segmenter.save_model(model_save_path)

        print(f"\n\n TIME TAKEN TO UPDATE SEGMENTER: {round((datetime.now() - start_update).seconds)} s with {len(oracle_results_desired_ids_list)} images for training")

        # evaluation 2: generate segmentations of validation and see how accurate our new segmenter is
        print("\n\n=== CREATING SEGMENTATIONS ON VALIDATION===")
        if unet:
            valid_input_dir =  f"/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/"
        else:
            valid_input_dir = os.path.join(
                os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task504_duke-mammo", 'imagesTs')
        
        valid_output_dir = os.path.join(run_dir_iter, f"ValSegmentations")
        validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
        print(f"\nMetric of new segmenter with '{oracle_query_method}' and {imgs_seen_list[index]} images seen in total is: {validation_metric}")

        print(f"# total images correctly labelled by segmenter: {len(correct_segmentations_ids)}")
        print(f"# total manually labelled images: {len(manually_labelled_ids)}")

        # Add results to results_dict
        results_dict[imgs_seen_list[index]] = [validation_metric, len(correct_segmentations_ids), model_save_path]

        # Save all patient scores so far for visualiation
        # TODO: Save the scores as dataframe, not pickled dictionary
        with open(os.path.join(run_dir, f"{oracle_query_method}_all_patient_scores_dict.pickle"), 'wb') as file:
            pickle.dump(all_patient_scores, file)

        # Save the filepaths of correct segmentations (can come from different AL iterations!)
        with open(os.path.join(run_dir, "segmenter_correct_segmentations_filepaths_list.pickle"), 'wb') as file:
            pickle.dump(correct_segmentations_filepaths, file)
    
        print("\n===ITERATION COMPLETED, MOVING ONTO NEXT ITER===\n\n\n")


    # TODO: Save experiment results incrementally so that even with errors, can retain results

    return results_dict


def run_active_learning_experiment(run_id, output_dir, random_seed, task_id=-1, unet=False):

    start = datetime.now()
    print("Starting run, and timing")

    # pandas dataframe where columns are query_type, query_number, IOU, # correctly labeled images by segmenter,
    # and location of saved model
    experiment_output = pd.DataFrame(
        columns=['random_seed', 'query_type', 'imgs_seen', 'IOU', 'num_correctly_labeled', 'saved_model_location'])


    # NOTE: In the current implementation, imgs seen must be divisible by 10
    # imgs_seen_list = [20, 35] 
    # imgs_seen_list = [20, 40, 60, 80]
    # imgs_seen_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    # imgs_seen_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    imgs_seen_list = list(range(20, 820, 20))
    imgs_seen_list.append(824)
    # imgs_seen_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65]
    # imgs_seen_list = [20]
    # oracle_query_methods = ["middle=0.5"]
    # oracle_query_methods = ["best"]
    # oracle_query_methods = ["best", "worst", "random", "best_worst_half"]
    # oracle_query_methods = ["random"]
    oracle_query_methods = ["random"]
    # oracle_query_methods = ["uniform", "random",
    #                         "percentile=0.8", "best", "worst"]
    # active_learning_initial(output_dir)
    
    # Print some information
    print("===INFORMATION ABOUT THE EXPERIMENT=== \n")
    exp_config_text = f"""Experiment date/time EST: {datetime.now()}
    Provided run_id: {run_id} \n
    Random seed: {random_seed} \n
    AL Query Methods: {oracle_query_methods} \n
    List of imgs seen: {imgs_seen_list} \n
    Initial batch size for discriminator: {BATCH_SIZE_INIT} \n
    Initial discriminator training # epochs: {INITIAL_DISCRIMINATOR_EPOCHS} \n
    Updating discriminator # epochs: {UPDATE_DISCRIMINATOR_EPOCHS} \n
    Update segmenter # epochs: {SEGMENTER_UPDATE_EPOCHS_INIT} \n
    STARTING TASK ID (-1 if default): {task_id} \n"""
    
    print(exp_config_text)
    
    # Save experiment configuration as a text file
    if os.path.exists(SAVE_DIR):
        print("The output directory already exists, so removing...")
        shutil.rmtree(SAVE_DIR)
        print("Done")
    os.mkdir(SAVE_DIR)
    with open(os.path.join(SAVE_DIR, "exp_configuration.txt"), "w") as file:
        file.write(exp_config_text)

    task_id_input = task_id

    for index, oracle_query_method in enumerate(oracle_query_methods):
        # Results in {imgs_seen: [validation metric, model_save_path]} dictionary format
        if task_id != -1:
            task_id_input = int(task_id)+len(imgs_seen_list)*index
        results = active_learning_experiment(
                                            imgs_seen_list,
                                            run_id,
                                            output_dir,
                                            oracle_query_method=oracle_query_method,
                                            random_seed=random_seed,
                                            given_task_id= task_id_input,
                                            unet=unet)
        
        for key in results.keys():
            experiment_output = experiment_output.append({'random_seed': random_seed,
                                                            'query_type': oracle_query_method,
                                                            'imgs_seen': key,
                                                            'IOU': results[key][0],
                                                            'num_correctly_labeled': results[key][1],
                                                            'saved_model_location': results[key][2]},
                                                            ignore_index=True)
    
    
    print("Finished run")
    print(f"Total time for AL: {datetime.now()-start}")
    print(f"RUN_ID: Run_{run_id}_{random_seed}")
    return experiment_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', "--run_id", required=True)
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("--random_seed", nargs=1, type=int, required=True)
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--nnunet', dest='unet', action='store_false')
    parser.set_defaults(unet=False)
    # Add task_id specification to run multiple experiments at once
    parser.add_argument('--task_id', nargs=1, type=int)
    parser.set_defaults(task_id=[-1])
    args = parser.parse_args()

    RANDOM_SEED_NUMBER = args.random_seed[0]
    torch.manual_seed(RANDOM_SEED_NUMBER)
    torch.cuda.manual_seed(RANDOM_SEED_NUMBER)
    np.random.seed(RANDOM_SEED_NUMBER)
    random.seed(RANDOM_SEED_NUMBER)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    # Task ID
    TASK_ID = args.task_id[0]

    RUN_ID = args.run_id
    OUTPUT_DIR = args.output_dir
    UNET = args.unet

    # Tunable Parameters
    BATCH_SIZE_INIT = 32
    INITIAL_DISCRIMINATOR_EPOCHS = 10
    UPDATE_DISCRIMINATOR_EPOCHS = 5
    SEGMENTER_UPDATE_EPOCHS_INIT = 5
    IOU_THRESHOLD = 0.7
    NUM_WORKERS = 2

    # DEFAULT PARAMETERS
    # BATCH_SIZE_INIT = 32
    # INITIAL_DISCRIMINATOR_EPOCHS = 10
    # UPDATE_DISCRIMINATOR_EPOCHS = 5
    # SEGMENTER_UPDATE_EPOCHS_INIT = 5
    # IOU_THRESHOLD = 0.9
    NUM_WORKERS = 2

    SAVE_DIR = os.path.join(OUTPUT_DIR, f"Run_{RUN_ID}_{RANDOM_SEED_NUMBER}")

    experiment_output = run_active_learning_experiment(RUN_ID, OUTPUT_DIR, RANDOM_SEED_NUMBER, TASK_ID, unet=UNET)
    os.makedirs(SAVE_DIR, exist_ok=True)
    experiment_output.to_csv(os.path.join(SAVE_DIR, 'experiment_output.csv'), sep=',')

    #TODO: Save script output to a file in the experiment directory

    # save the experiment output pandas dataframe

    # for i in range(len(metrics)):
    #     print(f"{query_numbers[i]} {metrics[i]}")
    # print("done")