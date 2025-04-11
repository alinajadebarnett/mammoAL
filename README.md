# Codebase for Active Learning Mammo Project - TODO


## Running an experiment
The relevant file for running an active learning experiment is experiment_v4.py. If you wish for an interactive setup, the notebook ActiveLearning.ipynb has cell-by-cell instructions.

Below is an example script to run an active learning experiment (using experiment_v4.py) with your desired parameters:

python experiment_v4.py --unet --task_id 336800 --run_id 1_11_random_44 --output_dir /usr/AllOracleRuns  --random_seed 44 --query_method "random"

### Relevant Parameters
task_id, run_id, output_dir: These parameters help set up the output directory, containing segmented images for each stage of active learning, as well as 
saved trained models. Each run creates a folder inside output_dir, which is named {task_id}_{run_id}.

query_method: Specifies the method used by the automatic oracle to self-query images as part of the active learning process. Query methods supported
are: "best", "worst", "random", and "percentile=0.x", where x is an integer from 1 to 9 inclusive. 

random_seed: Sets the random seed behind the active learning experiment

## Internally adjustable parameters
A few parameters are adjustable from inside the experiment_v4.py file, described below:

cbis_ddsm_dir: Located on line 48, this directory points to the CBIS-DDSM dataset, which is used to initially train the discriminator.
in_house_dir: Located on line 50, this directory points to your in-house dataset that your are labelling with this active learning process.
ground_truth_dir: Located on line 52, this directory points to the ground truth directory for the automatic oracle to reference.

 ## Packages and Dependencies
All required packages and their versions can be found in the requirements.txt file. We recommend creating a separate environment
(via Python or Conda), or using a dependency manager like Poetry.

Additionally, documentation for the nnUNet setup can be found at their repo: https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file. 
Relevant links are under the "How to Get Started" section.

