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

# Backend py file imports
from file_organization import save_active_learning_results, update_dir_with_oracle_info
from file_organization import redirect_saved_oracle_filepaths_to_thresheld_directory, remove_bad_oracle_results
from file_organization import save_files_for_nnunet
from dataloader import get_DataLoader
from disc_model import disc_model
from auto_oracle import query_oracle_automatic
import seg_model
import unet_model
import nnunet_model

def active_learning_experiment_initial(active_learning_train_cycles, imgs_seen, run_id, output_dir, iter_num, oracle_query_method, unet = False):
    # ACTIVE LEARNING STAGE

    # INITIALIZE CLASSIFIER
    # File definitions and static setup
    # files should be .npy, (2, , ) channels: image, binarized segmentation 
    discriminator_training_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    oracle_query_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    
    total_images_shown = 0
    saved_oracle_filepaths = []
    
    print("===TRAINING DISCRIMINATOR===")
    batch_size = 32
    dataloader = get_DataLoader(discriminator_training_dir, batch_size, 2)

    discriminator = disc_model()
    discriminator.load_model(dataloader)

    init_disc_epochs=10
    discriminator.initialize_model(batch_size=batch_size, epochs=init_disc_epochs) # initial training
    
    print("===LOADING INITIAL SEGMENTER (PRETRAINED ON CBIS-DDSM)===")
    run_dir = os.path.join(output_dir, run_id)
    segmentation_folder = '/usr/xtmp/gk122/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task504_duke-mammo/imagesTr'
    segmentation_save_dir = os.path.join(run_dir,"initial_segmentations")
    prev_segmentation_save_dir = "/usr/xtmp/sc834/mammoproj/nnunet_integration_tmp/AllOracleRuns/Run_23_06_19/Iter4/initial_segmentations"
    model_save_path = os.path.join(run_dir, "Iter" + str(iter_num)+".model")

    segmenter = nnunet_model.nnunet_model()
    segmenter.load_model("/usr/xtmp/sc834/mammoproj/data/nnUNet_preprocessed/Task501_cbis-ddsm")
    segmenter.save_model(model_save_path)

    if os.path.exists(segmentation_save_dir):
        os.rmdir(segmentation_save_dir)
    
    shutil.copytree(prev_segmentation_save_dir, segmentation_save_dir)

    print("===CONVERTING INITIAL SEGMENTATION PREDICTIONS INTO NUMPY)===")
    segmentations_score_dir = os.path.join(run_dir, "initial_segmentations_npy")

    if not os.path.exists(segmentations_score_dir):
        os.makedirs(segmentations_score_dir)
    
    # produces images of dimension (1, 640, 640)
    for filename in os.listdir(segmentation_save_dir):
        if 'DP' in filename and (str(filename).split('.')[-1] + '.npy') not in os.listdir(segmentation_save_dir):
            f = os.path.join(segmentation_save_dir, filename)
            img = sitk.GetArrayFromImage(sitk.ReadImage(f))
            np.save(os.path.join(run_dir,"initial_segmentations_npy", str(filename).split('.')[0] + ".npy"), img)
    
    # Try to restructure initial_segmentations_npy to follow Irregular, Round, Oval structure
    # NOTE: This solution should only be temporary - in real life, cannot structure output images into categories without the oracle.
    if not os.path.exists(os.path.join(segmentations_score_dir, "Irregular")):
        for name in ["Irregular", "Round", "Oval"]:
            path = os.path.join(segmentations_score_dir, name)

            if not os.path.exists(path):
                os.makedirs(path)

        all_filepaths = []
        for root, dirs, files in os.walk(ground_truth_dir):
                for file in files:
                    if file.endswith(".npy"):
                        all_filepaths.append(os.path.join(root, file))

        all_candidate_paths = []

        for root, dirs, files in os.walk(segmentations_score_dir):
            for file in files:
                filename = file.split('/')[-1]

                if '.npy' in filename:
                    all_candidate_paths.append(os.path.join(root, file))

        for path in all_candidate_paths:
            for candidate in all_filepaths:
                    if candidate.split('/')[-1] == path.split('/')[-1]:
                        os.rename(path, os.path.join(segmentations_score_dir, candidate.split('/')[-2], path.split('/')[-1]))

    print("===GENERATING INITIAL PATIENT SCORES===")
    discriminator_input_dir = update_dir_with_oracle_info_initial(save_dir=run_dir, mask_dir=segmentations_score_dir, im_dir=oracle_query_dir)
    discriminator_input_dir = os.path.join(run_dir, "OracleThresholdedImages_initial")

    segmentation_dataloader = get_DataLoader(discriminator_input_dir, batch_size, 2)
    patient_scores = discriminator.get_scores(segmentation_dataloader) # sorted

    # all_patient_scores = []
    # all_patient_scores.append(patient_scores)

    # QUERYING THE ORACLE
    print("===QUERYING THE ORACLE===")
    assert imgs_seen % 10 == 0
    active_learning_train_cycles = imgs_seen // 10
    # Begin loop over number of active learning/
    oracle_results = dict()
    oracle_results_thresholds = dict()
    for _ in tqdm(range(active_learning_train_cycles)):
        # Querying oracle - currently queries {query_cycles} times.
        try:
            oracle_results, oracle_results_thresholds = query_oracle_automatic(
                oracle_results, oracle_results_thresholds, patient_scores,
                ground_truth_dir, discriminator_input_dir,
                query_method=oracle_query_method, query_number=10)
        except Exception as e:
            print(str(e))
            print("Something went wrong with the automatic oracle query")
            sys.exit(1)
        total_images_shown += 10

        # Updating classifier 1 epoch at a time for 5 epochs.
        # print(f"=UPDATING DISCRIMINATOR (total images: {total_images_shown})")
        # for i in range(5):
        #     discriminator.update_model(oracle_results, batch_size=batch_size, num_epochs=1)

        #     patient_scores = discriminator.get_scores(segmentation_dataloader)
        #     all_patient_scores.append(patient_scores)

    # IN-BETWEEN STAGE
    print("===SAVING CLASSIFIED ORACLE RESULTS===")
    if not unet: 
        last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')))[-1]
        last_task = last_task.split('nnUNet_raw_data/Task')[-1][:3]
        task_id = int(last_task) + 1
    
    new_saved_oracle_filepaths = []

    for id in oracle_results:
        patID = id.split('/')[-1]

        for root, dirs, files in os.walk(oracle_query_dir):
            for filename in files:
                if patID in os.path.join(root, filename):
                    new_saved_oracle_filepaths.append(os.path.join(root, filename))
    
    save_files_for_nnunet(task_id, run_id, new_saved_oracle_filepaths)

    # SEGMENTATION STAGE
    # Preprocess data with information learned from active learning.
    # print("===PREPARING FOR SEGMENTATION STAGE===")
    # segmenter_train_dir = update_dir_with_oracle_info(run_dir, oracle_results_thresholds, oracle_query_dir)
    # new_saved_oracle_filepaths = redirect_saved_oracle_filepaths_to_thresheld_directory(
    #     saved_oracle_filepaths, segmenter_train_dir)
    # if not unet:
    #     last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')))[-1]
    #     last_task = last_task.split('nnUNet_raw_data/Task')[-1][:3]
    #     task_id = int(last_task) + 1
    #     save_files_for_nnunet(task_id, run_id, new_saved_oracle_filepaths)

    print("===UPDATING SEGMENTER FOR 5 EPOCHS===")
    # Train model using learned oracle data for 5 epochs
    # learned oracle data = images that are in the "new saved oracle filepaths" (the images that the oracle said looked good)
    if unet:
        segmenter = unet_model.unet_model()
        segmenter_train_dir = new_saved_oracle_filepaths
    else:
        segmenter = nnunet_model.nnunet_model()
        segmenter_train_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Task{task_id}_{run_id}')
    segmenter.load_model(segmenter_train_dir)
    segmenter.update_model(num_epochs=5)
    
    # potentially save model this iteration if we want. # to be used later 
    if unet:
        model_save_path = os.path.join(run_dir, "unetmodel.pth")
    else:
        model_save_path = os.path.join(run_dir, 'all', "Iter" + str(iter_num)+".model")
    segmenter.save_model(model_save_path)

    # # evaluation 1: generate new segmentations of training images and save them. (This is for the next stage of active learning)
    # # Evaluate ON IMAGES IN SEGMENTATION_FOLDER AND GENERATE SEGMENTATIONS OF THEM
    # print("=== CREATING SEGMENTATIONS FOR TRAIN SET ===")
    # # dir for marked correct by the oracle, do not overwrite the old segmentation, so save them here as an archive
    # correct_save_dir = os.path.join(run_dir, "Segmentations_C" )

    # # completely new set of segmentations created by the updated segmenter
    # save_dir = os.path.join(run_dir,"Segmentations")
    
    # if unet: 
    #     segmentation_folder = discriminator_training_dir
    # else:
    #     segmentation_folder = '/usr/xtmp/sc834/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task504_duke-mammo/imagesTr'
    # segmenter.predict(segmentation_folder, save_dir, correct_save_dir=correct_save_dir, saved_oracle_filepaths=saved_oracle_filepaths)   
    # Push save_dir as the oracle query dir for the next iteration. That's where we populate with unbinarized segmentations from recently trained segmenter

        
    # evaluation 2: generate segmentations of validation and see how accurate our new segmenter is
    print("=== CREATING SEGMENTATIONS FOR TEST SET ===")
    if unet:
        valid_input_dir =  f"/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/"
    else:
        valid_input_dir = os.path.join(
            os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task504_duke-mammo", 'imagesTs')
    
    valid_output_dir = os.path.join(run_dir, "ValSegmentations")
    validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
    print(f"Metric of new segmenter after active learning is: {validation_metric}.")
   
    return validation_metric, model_save_path

def active_learning_experiment_loop(active_learning_train_cycles, imgs_seen, run_id, output_dir, iter_num, oracle_query_method, unet = False):
    # ACTIVE LEARNING STAGE

    # INITIALIZE CLASSIFIER
    # File definitions and static setup
    # files should be .npy, (2, , ) channels: image, binarized segmentation 
    discriminator_training_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    oracle_query_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/final_dataset/train/"
    
    total_images_shown = 0
    saved_oracle_filepaths = []
    
    print("===TRAINING DISCRIMINATOR===")
    batch_size = 32
    dataloader = get_DataLoader(discriminator_training_dir, batch_size, 2)

    discriminator = disc_model()
    discriminator.load_model(dataloader)

    init_disc_epochs=10
    discriminator.initialize_model(batch_size=batch_size, epochs=init_disc_epochs) # initial training
    
    print("===LOADING INITIAL SEGMENTER (PRETRAINED ON CBIS-DDSM)===")
    run_dir = os.path.join(output_dir, run_id)
    segmentation_folder = '/usr/xtmp/gk122/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task504_duke-mammo/imagesTr'
    segmentation_save_dir = os.path.join(run_dir,"initial_segmentations")
    prev_segmentation_save_dir = "/usr/xtmp/sc834/mammoproj/nnunet_integration_tmp/AllOracleRuns/Run_23_06_19/Iter4/initial_segmentations"
    model_save_path = os.path.join(run_dir, "Iter" + str(iter_num)+".model")

    segmenter = nnunet_model.nnunet_model()
    segmenter.load_model("/usr/xtmp/sc834/mammoproj/data/nnUNet_preprocessed/Task501_cbis-ddsm")
    segmenter.save_model(model_save_path)

    if os.path.exists(segmentation_save_dir):
        os.rmdir(segmentation_save_dir)
    
    shutil.copytree(prev_segmentation_save_dir, segmentation_save_dir)

    print("===CONVERTING INITIAL SEGMENTATION PREDICTIONS INTO NUMPY)===")
    segmentations_score_dir = os.path.join(run_dir, "initial_segmentations_npy")

    if not os.path.exists(segmentations_score_dir):
        os.makedirs(segmentations_score_dir)
    
    # produces images of dimension (1, 640, 640)
    for filename in os.listdir(segmentation_save_dir):
        if 'DP' in filename and (str(filename).split('.')[-1] + '.npy') not in os.listdir(segmentation_save_dir):
            f = os.path.join(segmentation_save_dir, filename)
            img = sitk.GetArrayFromImage(sitk.ReadImage(f))
            np.save(os.path.join(run_dir,"initial_segmentations_npy", str(filename).split('.')[0] + ".npy"), img)
    
    # Try to restructure initial_segmentations_npy to follow Irregular, Round, Oval structure
    # NOTE: This solution should only be temporary - in real life, cannot structure output images into categories without the oracle.
    if not os.path.exists(os.path.join(segmentations_score_dir, "Irregular")):
        for name in ["Irregular", "Round", "Oval"]:
            path = os.path.join(segmentations_score_dir, name)

            if not os.path.exists(path):
                os.makedirs(path)

        all_filepaths = []
        for root, dirs, files in os.walk(ground_truth_dir):
                for file in files:
                    if file.endswith(".npy"):
                        all_filepaths.append(os.path.join(root, file))

        all_candidate_paths = []

        for root, dirs, files in os.walk(segmentations_score_dir):
            for file in files:
                filename = file.split('/')[-1]

                if '.npy' in filename:
                    all_candidate_paths.append(os.path.join(root, file))

        for path in all_candidate_paths:
            for candidate in all_filepaths:
                    if candidate.split('/')[-1] == path.split('/')[-1]:
                        os.rename(path, os.path.join(segmentations_score_dir, candidate.split('/')[-2], path.split('/')[-1]))

    print("===GENERATING INITIAL PATIENT SCORES===")
    discriminator_input_dir = update_dir_with_oracle_info_initial(save_dir=run_dir, mask_dir=segmentations_score_dir, im_dir=oracle_query_dir)
    discriminator_input_dir = os.path.join(run_dir, "OracleThresholdedImages_initial")

    segmentation_dataloader = get_DataLoader(discriminator_input_dir, batch_size, 2)
    patient_scores = discriminator.get_scores(segmentation_dataloader) # sorted

    # all_patient_scores = []
    # all_patient_scores.append(patient_scores)

    # QUERYING THE ORACLE
    print("===QUERYING THE ORACLE===")
    assert imgs_seen % 10 == 0
    active_learning_train_cycles = imgs_seen // 10
    # Begin loop over number of active learning/
    oracle_results = dict()
    oracle_results_thresholds = dict()
    for _ in tqdm(range(active_learning_train_cycles)):
        # Querying oracle - currently queries {query_cycles} times.
        try:
            oracle_results, oracle_results_thresholds = query_oracle_automatic(
                oracle_results, oracle_results_thresholds, patient_scores,
                ground_truth_dir, discriminator_input_dir,
                query_method=oracle_query_method, query_number=10)
        except Exception as e:
            print(str(e))
            print("Something went wrong with the automatic oracle query")
            sys.exit(1)
        total_images_shown += 10

        # Updating classifier 1 epoch at a time for 5 epochs.
        # print(f"=UPDATING DISCRIMINATOR (total images: {total_images_shown})")
        # for i in range(5):
        #     discriminator.update_model(oracle_results, batch_size=batch_size, num_epochs=1)

        #     patient_scores = discriminator.get_scores(segmentation_dataloader)
        #     all_patient_scores.append(patient_scores)

    # IN-BETWEEN STAGE
    print("===SAVING CLASSIFIED ORACLE RESULTS===")
    if not unet: 
        last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')))[-1]
        last_task = last_task.split('nnUNet_raw_data/Task')[-1][:3]
        task_id = int(last_task) + 1
    
    new_saved_oracle_filepaths = []

    for id in oracle_results:
        patID = id.split('/')[-1]

        for root, dirs, files in os.walk(oracle_query_dir):
            for filename in files:
                if patID in os.path.join(root, filename):
                    new_saved_oracle_filepaths.append(os.path.join(root, filename))
    
    save_files_for_nnunet(task_id, run_id, new_saved_oracle_filepaths)

    # SEGMENTATION STAGE
    # Preprocess data with information learned from active learning.
    # print("===PREPARING FOR SEGMENTATION STAGE===")
    # segmenter_train_dir = update_dir_with_oracle_info(run_dir, oracle_results_thresholds, oracle_query_dir)
    # new_saved_oracle_filepaths = redirect_saved_oracle_filepaths_to_thresheld_directory(
    #     saved_oracle_filepaths, segmenter_train_dir)
    # if not unet:
    #     last_task = sorted(glob.glob(os.path.join(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data','Task*')))[-1]
    #     last_task = last_task.split('nnUNet_raw_data/Task')[-1][:3]
    #     task_id = int(last_task) + 1
    #     save_files_for_nnunet(task_id, run_id, new_saved_oracle_filepaths)

    print("===UPDATING SEGMENTER FOR 5 EPOCHS===")
    # Train model using learned oracle data for 5 epochs
    # learned oracle data = images that are in the "new saved oracle filepaths" (the images that the oracle said looked good)
    if unet:
        segmenter = unet_model.unet_model()
        segmenter_train_dir = new_saved_oracle_filepaths
    else:
        segmenter = nnunet_model.nnunet_model()
        segmenter_train_dir = os.path.join(os.environ['nnUNet_preprocessed'], f'Task{task_id}_{run_id}')
    segmenter.load_model(segmenter_train_dir)
    segmenter.update_model(num_epochs=5)
    
    # potentially save model this iteration if we want. # to be used later 
    if unet:
        model_save_path = os.path.join(run_dir, "unetmodel.pth")
    else:
        model_save_path = os.path.join(run_dir, 'all', "Iter" + str(iter_num)+".model")
    segmenter.save_model(model_save_path)

    # # evaluation 1: generate new segmentations of training images and save them. (This is for the next stage of active learning)
    # # Evaluate ON IMAGES IN SEGMENTATION_FOLDER AND GENERATE SEGMENTATIONS OF THEM
    # print("=== CREATING SEGMENTATIONS FOR TRAIN SET ===")
    # # dir for marked correct by the oracle, do not overwrite the old segmentation, so save them here as an archive
    # correct_save_dir = os.path.join(run_dir, "Segmentations_C" )

    # # completely new set of segmentations created by the updated segmenter
    # save_dir = os.path.join(run_dir,"Segmentations")
    
    # if unet: 
    #     segmentation_folder = discriminator_training_dir
    # else:
    #     segmentation_folder = '/usr/xtmp/sc834/mammoproj/data/nnUNet_raw_data_base/nnUNet_raw_data/Task504_duke-mammo/imagesTr'
    # segmenter.predict(segmentation_folder, save_dir, correct_save_dir=correct_save_dir, saved_oracle_filepaths=saved_oracle_filepaths)   
    # Push save_dir as the oracle query dir for the next iteration. That's where we populate with unbinarized segmentations from recently trained segmenter

        
    # evaluation 2: generate segmentations of validation and see how accurate our new segmenter is
    print("=== CREATING SEGMENTATIONS FOR TEST SET ===")
    if unet:
        valid_input_dir =  f"/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/"
    else:
        valid_input_dir = os.path.join(
            os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', f"Task504_duke-mammo", 'imagesTs')
    
    valid_output_dir = os.path.join(run_dir, "ValSegmentations")
    validation_metric = segmenter.validate(valid_input_dir, valid_output_dir)
    print(f"Metric of new segmenter after active learning is: {validation_metric}.")
   
    return validation_metric, model_save_path

# Gets the number of current run
# Indexing starts at 0; 0 is initial training and segmentations
def get_iter(output_dir):
    if os.path.isdir(output_dir):
        iter_list = [f.name for f in os.scandir(output_dir)]
        iter_list = [f for f in iter_list if "Iter" in f]
        iter_list.sort()
        return int(iter_list[len(iter_list) - 1][4:]) + 1
    else:
        return 0

def run_active_learning_experiment(run_id, output_dir, random_seed, unet=False):
    # ADDED
    start = datetime.now()
    print("Starting run, and timing")

    # pandas dataframe where columns are query_type query_number IOU location of saved model
    experiment_output = pd.DataFrame(
        columns=['random_seed', 'query_type', 'imgs_seen', 'IOU', 'saved_model_location'])

    # imgs_seen_list = [20]
    imgs_seen_list = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    oracle_query_methods = ["worst"]
    # oracle_query_methods = ["uniform", "random",
    #                         "percentile=0.8", "best", "worst"]

    output_dir = os.path.join(output_dir, f"Run_{run_id}_{random_seed}")
    iter_num = get_iter(output_dir)
    global iter_dir
    iter_dir = os.path.join(output_dir, f"Iter{iter_num}")
    os.makedirs(iter_dir)

    for oracle_query_method in oracle_query_methods:
        for imgs_seen in imgs_seen_list:
            run_unique_id = f"{oracle_query_method}_{imgs_seen}"
            print(run_unique_id)
            if iter_num == 0:
                validation_metric, saved_model_location = active_learning_experiment_initial(10,
                                                                    imgs_seen,
                                                                    run_unique_id,
                                                                    iter_dir,
                                                                    iter_num,
                                                                    oracle_query_method=oracle_query_method,
                                                                    unet=unet)
            else:
                validation_metric, saved_model_location = active_learning_experiment_loop(10,
                                                                    imgs_seen,
                                                                    run_unique_id,
                                                                    iter_dir,
                                                                    iter_num,
                                                                    oracle_query_method=oracle_query_method,
                                                                    unet=unet)

            print(
                f"Done with {imgs_seen} for query method {oracle_query_method}")
            experiment_output = experiment_output.append({'random_seed': random_seed,
                                                          'query_type': oracle_query_method,
                                                          'imgs_seen': imgs_seen,
                                                          'IOU': validation_metric,
                                                          'saved_model_location': saved_model_location},
                                                          ignore_index=True)

    print("Finished run")
    print(datetime.now()-start)
    return experiment_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', "--run_id", required=True)
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("--random_seed", nargs=1, type=int, required=True)
    parser.add_argument('--unet', action='store_true')
    parser.add_argument('--nnunet', dest='unet', action='store_false')
    parser.set_defaults(unet=False)
    args = parser.parse_args()

    random_seed = args.random_seed[0]
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    run_id = args.run_id
    output_dir = args.output_dir
    unet = args.unet

    experiment_output = run_active_learning_experiment(run_id, output_dir, random_seed, unet=unet)
    experiment_output.to_csv(os.path.join(iter_dir, 'experiment_output.csv'), sep=',')
    
    # save_dir = os.path.join(output_dir, f"Run_{run_id}/")
    # os.makedirs(save_dir, exist_ok=True)
    # experiment_output.to_csv(os.path.join(save_dir, 'experiment_output.csv'), sep=',')

    # save the experiment output pandas dataframe

    # for i in range(len(metrics)):
    #     print(f"{query_numbers[i]} {metrics[i]}")
    # print("done")
