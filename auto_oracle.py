import numpy as np
import torch
import random

from torchvision import transforms

from dataloader import get_DataLoader

def get_ints_torch(mask):
    return torch.where(mask>0.2,1,0)


def get_binary_mask_threshold_torch(mask,threshold):
    return torch.where(mask > threshold, 1, 0)


def intersection_over_union(output_mask,ground_mask):
    ground_mask = get_ints_torch(ground_mask).squeeze(1)
    summed = ground_mask + output_mask
    twos = summed - 2
    # num = 256*256 - torch.count_nonzero(twos)
    num = torch.sum(summed > 1)
    denom = torch.count_nonzero(summed)
    outputs = torch.div(num,denom)
    return torch.mean(outputs)

#TODO: Check if patID has shape subdir. If not, you're going to have to find them manually.
def calculate_iou(patID,threshold,ground_truth_dir,segmentation_dir):

    # ADDED:
    newPatID = patID
    if str(patID)[0] != '/':
        newPatID = '/' + patID

    ground_truth_path = ground_truth_dir + newPatID + ".npy"
    segmentation_path = segmentation_dir + newPatID + ".npy"

    # should segmentation [1,:,:] instead be [0,:,:]
    ground_truth = np.load(ground_truth_path)[1,:,:]
    segmentation = np.load(segmentation_path)[1,:,:]
    
    transforms_arr = [transforms.ToTensor(),transforms.Resize((256,256))]
    # transforms_arr = [transforms.ToTensor()]
    image_transform = transforms.Compose(transforms_arr)
    
    ground_truth = image_transform(ground_truth)[0,:,:]
    segmentation = image_transform(segmentation)[0,:,:]
    
    binarized = get_binary_mask_threshold_torch(segmentation,0)

    return intersection_over_union(binarized,ground_truth)

#Mimics same input/output as ask_oracle from oracle.py. 
#Only difference is that it asks for two directories as it needs both the unbinarized seg dir and the ground truth dir
def ask_oracle_automatic(oracle_results, oracle_results_thresholds,oracle_queries, ground_truth_dir,segmentation_dir, iou_threshold=0.9):
    for patID in oracle_queries:
        max_iou = -1
        max_threshold = 0
        #calculate iou over a variety of thresholds
        thresholds = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
        iou = calculate_iou(patID,iou_threshold,ground_truth_dir,segmentation_dir)
        for threshold in thresholds:
            try:
                iou = calculate_iou(patID,threshold,ground_truth_dir,segmentation_dir)
            except:
                return None,None
            # Check if proposed segmentation is very close to ground truth (Starting off at 0.2)
            if iou > max_iou:
                max_iou = iou
                max_threshold = threshold
        if(max_iou > iou_threshold):
            #return a 1 with correct threshold
            oracle_results[patID] = 1
        else:
            oracle_results[patID] = 0
        oracle_results_thresholds[patID] = max_threshold #Regardless of the correctness of segmentation, we assign the threshold which gives the highest IOU.
    return oracle_results,oracle_results_thresholds


def query_oracle_automatic(oracle_results,oracle_results_thresholds,patient_scores,ground_truth_dir,segmentation_dir,query_method="uniform",query_number=10):
    if query_number==0:
        print("Why are you asking for 0 queries?")
        return oracle_results
    if query_number>len(patient_scores):
        print("Query too big for number of patients")
        return oracle_results
    oracle_queries = []
    patient_scores_minus_oracle_results = []
    for patient_score in list(patient_scores.keys()):
        if patient_score not in list(oracle_results.keys()):
            patient_scores_minus_oracle_results.append(patient_score)

    if query_method=="uniform":
        step = len(patient_scores_minus_oracle_results) // query_number
        start = 0 if step == 1 else step
        for i in range(start, len(patient_scores_minus_oracle_results), step):
            oracle_queries.append(patient_scores_minus_oracle_results[i])
            if len(oracle_queries) == query_number:
                break
    elif query_method=="random":
        indices = random.sample(list(np.arange(len(patient_scores_minus_oracle_results))), query_number)
        for i in indices:
                oracle_queries.append(patient_scores_minus_oracle_results[i])
    elif query_method=="best":
         for i in range(query_number):
            oracle_queries.append(patient_scores_minus_oracle_results[len(patient_scores_minus_oracle_results)-i-1])
    elif query_method=="worst":
        for i in range(query_number):
                oracle_queries.append(patient_scores_minus_oracle_results[i])
    elif 'middle' in query_method:
        #find the number of elements closest to 0.5
        split_val = float(query_method.split('=')[-1])
        middle_index = int(len(patient_scores_minus_oracle_results)/(1/split_val))
        left_bound = 0 if middle_index - int(query_number/2) < 0 else middle_index - int(query_number/2)
        indices = list(range(middle_index,middle_index+int(query_number/2))) + range(left_bound,middle_index)
        for i in indices:
                oracle_queries.append(patient_scores_minus_oracle_results[i])
        print("Debugging for middle index: "  + str(middle_index) + " " + str(patient_scores[middle_index]))
    elif "percentile" in query_method:
        percentile = float(query_method.split('=')[-1])
        near_index = int(len(patient_scores_minus_oracle_results) * percentile)
        indices = list(range(near_index - int(query_number/2), near_index)) + list(range(near_index, near_index + int(query_number/2)))
        for i in indices:
            oracle_queries.append(patient_scores_minus_oracle_results[i])
    else:
        print("You entered an unsupported query method.")
        return oracle_results,oracle_results_thresholds

    # if query_method=="uniform":
    #     step = len(patient_scores_minus_oracle_results) // query_number
        # start = 0 if step == 1 else step
        # for i in range(start, len(patient_scores_minus_oracle_results), step):
        #     oracle_queries.append(patient_scores_minus_oracle_results[i])
        #     if len(oracle_queries) == query_number:
        #         break
                
    # elif query_method=="random":
    #     indices = random.sample(len(patient_scores), query_number)
    #     for i in indices:
    #         if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
    #             oracle_queries.append(list(patient_scores.keys())[i])
    # elif query_method=="best":
    #      for i in range(query_number-1,-1,-1):
    #          if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
    #             oracle_queries.append(list(patient_scores.keys())[i])
    # elif query_method=="worst":
    #     for i in range(query_number):
    #         if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
    #             oracle_queries.append(list(patient_scores.keys())[i])
    # elif 'middle' in query_method:
    #     #find the number of elements closest to 0.5
    #     split_val = float(query_method.split('=')[-1])
    #     middle_index = int(len(patient_scores.keys())/(1/split_val))
    #     left_bound = 0 if middle_index - int(query_number/2) < 0 else middle_index - int(query_number/2)
    #     indices = list(range(middle_index,middle_index+int(query_number/2))) + range(left_bound,middle_index)
    #     for i in indices:
    #         if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
    #             oracle_queries.append(list(patient_scores.keys())[i])
    #     print("Debugging for middle index: "  + str(middle_index) + " " + str(patient_scores[middle_index]))
    # elif "percentile" in query_method:
    #     percentile = float(query_method.split('=')[-1])
    #     near_index = int(len(patient_scores.keys()) * percentile)
    #     indices = list(range(near_index - int(query_number/2), near_index)) + list(range(near_index, near_index + int(query_number/2)))
    #     for i in indices:
    #         if list(patient_scores.keys())[i] not in list(oracle_results.keys()):
    #             oracle_queries.append(list(patient_scores.keys())[i])
    # else:
    #     print("You entered an unsupported query method.")
    #     return oracle_results,oracle_results_thresholds

    oracle_results, oracle_results_thresholds = ask_oracle_automatic(oracle_results,oracle_results_thresholds,oracle_queries,ground_truth_dir,segmentation_dir)
    # if oracle_results is None:
    #     return None,None
    return oracle_results, oracle_results_thresholds

# Added to take into account nnUNet produces binarized predictions
def ask_oracle_automatic_nnunet(oracle_results, oracle_queries, ground_truth_dir, segmentation_dir, iou_threshold=0.9):
    for patID in oracle_queries:
        iou = calculate_iou(patID,iou_threshold,ground_truth_dir,segmentation_dir)

        if(iou > iou_threshold):
            #return a 1 with correct threshold
            oracle_results[patID] = 1
        else:
            oracle_results[patID] = 0
        
    return oracle_results

def query_oracle_automatic_no_threshold(oracle_results,patient_scores,ground_truth_dir,segmentation_dir,query_method="uniform",query_number=10, iou_threshold=0.9):
    if query_number==0:
        print("Why are you asking for 0 queries?")
        return oracle_results
    if query_number>len(patient_scores):
        print("Query too big for number of patients")
        return oracle_results
    oracle_queries = []
    patient_scores_minus_oracle_results = []
    for patient_score in list(patient_scores.keys()):
        if patient_score not in list(oracle_results.keys()):
            patient_scores_minus_oracle_results.append(patient_score)

    if len(patient_scores_minus_oracle_results) < query_number:
        query_number = len(patient_scores_minus_oracle_results)

    if query_method=="uniform":
        step = len(patient_scores_minus_oracle_results) // query_number
        start = 0 if step == 1 else step
        for i in range(start, len(patient_scores_minus_oracle_results), step):
            oracle_queries.append(patient_scores_minus_oracle_results[i])
            if len(oracle_queries) == query_number:
                break
    elif query_method=="random":
        indices = random.sample(list(np.arange(len(patient_scores_minus_oracle_results))), query_number)
        for i in indices:
            oracle_queries.append(patient_scores_minus_oracle_results[i])
    elif query_method=="best":
         for i in range(query_number):
            oracle_queries.append(patient_scores_minus_oracle_results[len(patient_scores_minus_oracle_results)-i-1])
    elif query_method=="worst":
        for i in range(query_number):
            oracle_queries.append(patient_scores_minus_oracle_results[i])

    # NOTE: Experiment - try mixing best and worst
    elif query_method=="best_worst_half":
        for i in range(query_number//2):
            oracle_queries.append(patient_scores_minus_oracle_results[i])
        for i in range(query_number//2):
            oracle_queries.append(patient_scores_minus_oracle_results[len(patient_scores_minus_oracle_results)-i-1])

    elif query_method=="middle":
        #find the number of elements closest to 0.5
        # split_val = float(query_method.split('=')[-1])
        # print(f'length of results: {len(patient_scores_minus_oracle_results)}')
        middle_index = int(len(patient_scores_minus_oracle_results)/2)
        # print(f'middle_index: {middle_index}')
        left_bound = 0 if middle_index - int(query_number/2) < 0 else middle_index - int(query_number/2)
        # print(f'left bound: {left_bound}')
        indices = list(range(middle_index,middle_index+int(query_number/2))) + list(range(left_bound,middle_index))
        # print(f'indices: {indices}')
        for i in indices:
            oracle_queries.append(patient_scores_minus_oracle_results[i])

    elif "percentile" in query_method:
        percentile = float(query_method.split('=')[-1])
        scores_len = len(patient_scores_minus_oracle_results)
        print(f'length of remaining results: {scores_len}')

        near_index = int(scores_len * percentile)
        print(f'near_index: {near_index}')

        if near_index - int(query_number/2) < 0:
            if percentile<=0.5:
                indices = list(range(0, query_number))
            else:
                indices = list(range(scores_len-query_number, scores_len))
        elif near_index + int(query_number/2) > scores_len:
            if percentile<=0.5:
                indices = list(range(0, query_number))
            else:
                indices = list(range(scores_len-query_number, scores_len))
        else:
            indices = list(range(near_index - int(query_number/2), near_index)) + list(range(near_index, near_index + int(query_number/2)))
        
        print(f'indices: {indices}')
        for i in indices:
            oracle_queries.append(patient_scores_minus_oracle_results[i])
    else:
        print("You entered an unsupported query method.")
        return oracle_results

    # oracle_results, oracle_results_thresholds = ask_oracle_automatic(oracle_results,oracle_results_thresholds,oracle_queries,ground_truth_dir,segmentation_dir)
    oracle_results = ask_oracle_automatic_nnunet(oracle_results, oracle_queries, ground_truth_dir, segmentation_dir, iou_threshold)
    # if oracle_results is None:
    #     return None,None
    # return oracle_results, oracle_results_thresholds
    return oracle_results

# Added to allow simple oracle of ground_truth images
def query_oracle_automatic_ground_truth(oracle_results, patient_scores,ground_truth_dir,segmentation_dir,query_method="uniform",query_number=10):
    if query_number==0:
        print("Why are you asking for 0 queries?")
        return oracle_results
    if query_number>len(patient_scores):
        print("Query too big for number of patients")
        return oracle_results
    
    oracle_temp = oracle_results

    patient_scores_minus_oracle_results = []
    for patient_score in list(patient_scores.keys()):
        if patient_score not in list(oracle_results.keys()):
            patient_scores_minus_oracle_results.append(patient_score)
    
    if query_method=="uniform":
        step = len(patient_scores_minus_oracle_results) // query_number
        start = 0 if step == 1 else step
        for i in range(start, len(patient_scores_minus_oracle_results), step):
            oracle_queries.append(patient_scores_minus_oracle_results[i])
            if len(oracle_queries) == query_number:
                break
    elif query_method=="random":
        indices = random.sample(list(np.arange(len(patient_scores_minus_oracle_results))), query_number)
        for i in indices:
            oracle_temp[patient_scores_minus_oracle_results[i]] = 1
    elif query_method=="best":
         for i in range(query_number):
            oracle_temp[patient_scores_minus_oracle_results[len(patient_scores_minus_oracle_results)-i-1]] = 1
    elif query_method=="worst":
        for i in range(query_number):
            oracle_temp[patient_scores_minus_oracle_results[i]] = 1
    

    return oracle_temp


# if __name__=="__main__":
#     model = None #Replace model with trained classifier for running experiments on active learning.
#     classifier_training_dir = "/usr/xtmp/vs196/mammoproj/Data/manualfa/train/" #Manually labelled train data. Not sure if we use this or validation dir.
#     dataloader = get_DataLoader(classifier_training_dir,32,2) 
#     oracle_results = dict()
#     oracle_results_thresholds = dict()
#     patient_scores = get_patient_scores(model,dataloader)
#     ground_truth_dir = "/usr/xtmp/vs196/mammoproj/Data/manualfa/manual_validation/" #manual segmentations
#     segmentation_dir = "/usr/xtmp/vs196/mammoproj/Data/manualfa/validation_histeq_ff/" #Not control. This segs from unet trained after one round of active learning (with flood fill).
#     #We may have to make a separate method to generate new segmentations based on different unet models.
#     oracle_results, oracle_results_thresholds = query_oracle_automatic(oracle_results,oracle_results_thresholds,patient_scores,ground_truth_dir,segmentation_dir,query_method="uniform",query_number=10)