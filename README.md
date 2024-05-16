## File Description :

1. Parkinsson Disease.csv : Dataset 1 with 22 features.
2. pd_speech_features.csv : Dataset 2 with 753 features.
3. Model_1st_Dataset.ipynb : Contains the unique bagging model trained on dataset 1 [Note: Used MRMR as feature selection ]
4. Model_2nd_Dataset.ipynb : Contains the unique bagging model trained on dataset 2 [Note: Used MRMR as feature selection ]
5. model_testing.ipynb : Contains the metrics of the classifier models, without FS, that are used in the bagging classifier. 
6. audio_extractor.py : A Streamlit application which basically process the audio file you upload. [Check it out here](https://audio-extractor.streamlit.app/)
7. pearson_selection_resampling.ipynb : Contains the code of the new model [ New feature selection + sample resampling + new bagging model where the bags are based on the samples ]
8. FS_classif.ipynb : Contains the metrics of the classifier models, with FS, that are used in the bagging classifier.
9. Metrics@Dataset2@Btech.xlsx : Contains the tables/metrics which will be required for the results section.

## Few points to note:

1. Files 3 and 4 are not required actually but for understanding the classifier model you may refer it. Note that here the classification model includes feature resampling and mrmr as feature selection method. 
2. Currently the files 5,7,8 contains "pd_speech_features.csv" as the dataset.
3. To use "Parkinson Disease.csv" simply replace the dataset and replace 'id' with 'name', 'class' with 'status'. 
