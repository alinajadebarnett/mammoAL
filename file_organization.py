import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet_model import convert_2d_image_to_nifti, plan_and_preprocess
from floodfill import convert_directory_to_floodfill
from manual_oracle import  save_oracle_results, save_oracle_results_all

import SimpleITK as sitk

from skimage.transform import resize


# ADDED
def convert_initial_segmentations_to_numpy(im_dir, save_dir):
    # Convert all images within the im_dir from nii.gz to numpy, and save them in save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # produces images of dimension (1, 640, 640)
    for filename in tqdm(os.listdir(im_dir)):
        if 'nii.gz' in filename and (str(filename).split('.')[-1] + '.npy') not in os.listdir(im_dir):
            f = os.path.join(im_dir, filename)
            img = sitk.GetArrayFromImage(sitk.ReadImage(f))
            np.save(os.path.join(save_dir, str(filename).split('.')[0] + ".npy"), img)

# ADDED
def change_image_paths_to_follow_file_structure(im_dir, reference_dir):
    # NOTE: Temporary solution only - in real life, cannot structure output images into categories without the oracle.
    # Re-path images in a folder to follow Irregular/ Oval/ Round/ structure, provided by the reference_dir

    if not os.path.exists(os.path.join(im_dir, "Irregular")):
        for name in ["Irregular", "Round", "Oval"]:
            path = os.path.join(im_dir, name)

            if not os.path.exists(path):
                os.makedirs(path)

        all_filepaths_reference = []
        for root, dirs, files in os.walk(reference_dir):
                for file in files:
                    if file.endswith(".npy"):
                        all_filepaths_reference.append(os.path.join(root, file))

        all_candidate_paths = []

        for root, dirs, files in os.walk(im_dir):
            for file in files:
                filename = file.split('/')[-1]

                if '.npy' in filename:
                    all_candidate_paths.append(os.path.join(root, file))

        for path in all_candidate_paths:
            for candidate in all_filepaths_reference:
                    if candidate.split('/')[-1] == path.split('/')[-1]:
                        os.rename(path, os.path.join(im_dir, candidate.split('/')[-2], path.split('/')[-1]))



def get_binary_mask_threshold(mask,threshold):
    return np.where(mask > threshold, 1, 0)


#Removes all 0's from oracle_results (images that oracle said are incorrect)
def remove_bad_oracle_results(oracle_results):
    output = {}
    for patient in oracle_results.keys():
        if oracle_results[patient]==1:
            output[patient] = oracle_results[patient]
    return output


# Saves any correct segmentations found by the oracle, and additionally pickle dumps any structures
def save_active_learning_results(save_dir, oracle_results, oracle_results_thresholds, im_dir):
    correct_segs_save_dir = os.path.join(save_dir, "CorrectSegmentations")

    saved_oracle_filepaths = save_oracle_results(
        oracle_results, oracle_results_thresholds, im_dir, correct_segs_save_dir)
    fpath = os.path.join(save_dir, "saved_data_struct")
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    saved_oracle_filepaths_filepath = os.path.join(fpath, "Oracle_Filepaths.pickle")
    pickle.dump(saved_oracle_filepaths, open(saved_oracle_filepaths_filepath, "wb"))
    pickle.dump(oracle_results, open(os.path.join(fpath, "Oracle_Results.pickle"), "wb"))
    pickle.dump(oracle_results_thresholds, open(os.path.join(fpath,"Oracle_Results_Thresholds.pickle"), "wb"))

    return saved_oracle_filepaths

# Saves any correct segmentations found by the oracle, and additionally pickle dumps any structures
def save_active_learning_results_all(save_dir, oracle_results, oracle_results_thresholds, im_dir, ground_truth_dir):
    correct_segs_save_dir = os.path.join(save_dir, "CorrectSegmentations")

    good_filepaths, bad_filepaths = save_oracle_results_all(
        oracle_results, oracle_results_thresholds, im_dir, correct_segs_save_dir, ground_truth_dir)
    fpath = os.path.join(save_dir, "saved_data_struct")
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    saved_oracle_filepaths_filepath = os.path.join(fpath, "Oracle_Filepaths.pickle")
    pickle.dump(good_filepaths, open(saved_oracle_filepaths_filepath, "wb"))
    pickle.dump(bad_filepaths, open(saved_oracle_filepaths_filepath, "wb"))
    pickle.dump(oracle_results, open(os.path.join(fpath, "Oracle_Results.pickle"), "wb"))
    pickle.dump(oracle_results_thresholds, open(os.path.join(fpath,"Oracle_Results_Thresholds.pickle"), "wb"))

    return good_filepaths, bad_filepaths


#Applies the threshold to each mask saved from the oracle. Resaves [arr,thresholded_mask] stack into save_dir using same conventions (.../Shape/name.npy)
def threshold_and_save_images(saved_oracle_filepaths, oracle_results_thresholds, save_dir):
    for filepath in tqdm(saved_oracle_filepaths):
        if ("/".join(filepath.split("/")[-2:]))[:-4] not in oracle_results_thresholds:
            threshold = 0.2
        else:
            threshold = oracle_results_thresholds[("/".join(filepath.split("/")[-2:]))[:-4]]
        arr_and_mask = np.load(filepath)
        copy_arr_mask = arr_and_mask.copy() 

        arr = copy_arr_mask[0,:,:].copy()
        mask = copy_arr_mask[1,:,:].copy()
        #apply threshold to mask
        mask = get_binary_mask_threshold(mask, threshold)
        to_save = np.stack([arr, mask])
        
        save_save_dir = os.path.join(save_dir, "/".join(filepath.split("/")[-2:]))
        if not os.path.exists(os.path.join(save_dir, filepath.split("/")[-2])):
            os.makedirs(os.path.join(save_dir, filepath.split("/")[-2]))
        np.save(save_save_dir, to_save)

# ADDED to save original image and predicted mask pair for input to the discriminator
def save_image_mask_pairs_initial(saved_oracle_filepaths, mask_filepaths, save_dir):
    for filepath in tqdm(saved_oracle_filepaths):

        mask_filepath = ''
        filename = filepath.split('/')[-1]
        
        # Finds matching patient ID for mask prediction. 
        for path in mask_filepaths:
            if path.split('/')[-1] == filename:
                mask_filepath = path

        arr_original = np.load(filepath)

        # Arr is of shape (224, 224)
        arr = arr_original[0,:,:]

        if mask_filepath != '':
            mask = np.load(mask_filepath)

            # Reduce/reshape the size of the mask to (224, 224) from (1, 640, 640)
            # resized_mask = cv2.resize(mask[0,:,:], (224, 224), interpolation=cv2.INTER_NEAREST)
            resized_mask = resize(mask[0,:,:], (224, 224), order=0, preserve_range=True)

            # Creates (2, 224, 224) stack
            to_save = np.stack([arr, resized_mask])

            save_save_dir = os.path.join(save_dir, "/".join(filepath.split("/")[-2:]))
            if not os.path.exists(os.path.join(save_dir, filepath.split("/")[-2])):
                os.makedirs(os.path.join(save_dir, filepath.split("/")[-2]))
            np.save(save_save_dir, to_save)
        else:
            print(f"{filename} NOT MATCHED TO A MASK, MOVING ON")
            pass
        

def update_dir_with_oracle_info(save_dir, oracle_results_thresholds, im_dir):
    save_dir = os.path.join(save_dir, "OracleThresholdedImages")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # find all filepaths in im_dir
    all_filepaths = []
    for root, dirs, files in os.walk(im_dir):
        for file in files:
            if file.endswith(".npy"):
                all_filepaths.append(os.path.join(root, file))

    threshold_and_save_images(
        all_filepaths, oracle_results_thresholds, save_dir)
    save_dir = convert_directory_to_floodfill(save_dir, iter0=False)
    return save_dir

# ADDED to save original image and predicted mask pair for input to the discriminator
def update_dir_with_oracle_info_initial(save_dir, im_dir, mask_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # find all filepaths in im_dir
    img_filepaths = []
    for root, dirs, files in os.walk(im_dir):
        for file in files:
            if file.endswith(".npy"):
                img_filepaths.append(os.path.join(root, file))

    mask_filepaths = []
    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(".npy"):
                mask_filepaths.append(os.path.join(root, file))

    # Uses custom function to stack original image and predicted masks from the segmenter.
    save_image_mask_pairs_initial(
        img_filepaths, mask_filepaths, save_dir)
    
    ## Is floodfill necesssary?
    # save_dir = convert_directory_to_floodfill(save_dir, iter0=False)
    return save_dir


def redirect_saved_oracle_filepaths_to_thresheld_directory(saved_oracle_filepaths, im_dir):
    new_filepaths = [(im_dir + "/".join(filepath.split("/")[-2:]))
                     for filepath in saved_oracle_filepaths]
    return new_filepaths


def save_files_for_nnunet(task_id, run_id, filepaths):
    new_gz_dir = os.path.join(os.environ['nnUNet_raw_data_base'],'nnUNet_raw_data', f'Task{task_id}_{run_id}')
    os.makedirs(new_gz_dir, exist_ok=True)

    target_imagesTr = os.path.join(new_gz_dir, 'imagesTr')
    target_labelsTr = os.path.join(new_gz_dir, 'labelsTr')
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)

    for t in filepaths:
        unique_name = os.path.splitext(os.path.split(t)[-1])[0]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_file = t

        img = np.load(input_file)
        img = img.copy()

        img_r = cv2.resize(img[0], (640,640))
        mask = cv2.resize(img[1], (640,640))

        output_image_file = os.path.join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = os.path.join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you
        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please take a look at the code for this function and adapt it to your needs
        train_img = convert_2d_image_to_nifti(img_r.copy(), output_image_file, is_seg=False)

        # nnU-Net expects the labels to be consecutive integers. This can be achieved with setting a transform
        train_seg = convert_2d_image_to_nifti(mask.copy(), output_seg_file, is_seg=True,
                                    transform=lambda x: (x >= 1).astype(int))

        # finally we can call the utility for generating a dataset.json
    generate_dataset_json(os.path.join(new_gz_dir, 'dataset.json'), target_imagesTr, None, ("RGB",),
                    labels={0: 'background', 1: 'lesion'}, 
                    dataset_name=f'Task{task_id}_{run_id}', 
                    license='hands off!')
    
    # subprocess.run(["nnUNet_plan_and_preprocess", "-t", f"{task_id}", "--verify_dataset_integrity"])
    plan_and_preprocess([task_id, ], verify_integrity = True)