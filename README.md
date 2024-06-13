# Comparison of Feature Selection and Supervised Methods for Classifying Gait Disorders

### configuration for data preparation
- set the directories of csv file, source and target files to the config_data.yaml file in config folder
- the mode parameter can be set for detection or tracking mode 


### Manual feature extraction
- check if there are data files in its directory 
- 1) to get skeleton feature and some data preprocessing steps run: python prepare_dataset/manual_feat/1_skeleton_feat.py <br />
- note: if both tracking and detection data needed, it need to manually change the mode value in config_data and sepratly run first step <br />
- 2) to generate 70 featurres run: python prepare_dataset/manual_feat/2_generated_70_feat.py <br />
- 3) to generate data with all different feature sizes run: python prepare_dataset/manual_feat/3_make_diff_feature_size.py <br />


### Best features selection
- 1) to generate best feature indexes run: python prepare_dataset/best_features_selection/1_best_feature_selections.py <br />
- 2) to generate best feature selection data run: python prepare_dataset/best_features_selection/2_apply_best_features_to_dataset.py <br />


### configuration for train model
- set parameters to the config_train.yaml file in config folder <br />


### train 
- check there are Xtrain.File, Xtest.File, ytrain.File, ytest.File in configured directories in data folder <br />
- run: python src/train.py <br />

### test trained models
- trained model should be in saved_models folder <br />
- run: python src/test.py <br />
- Note: the list of parameters in config_train file for models, feautre sizes should be same for train and test

