## File Description :

1. Parkinsson Disease.csv : Dataset 1 with 22 features.
2. pd_speech_features.csv : Dataset 2 with 753 features.
3. Model_1st_Dataset.ipynb : Contains the unique bagging model trained on dataset 1 [Note: Used MRMR as feature selection ]
4. Model_2nd_Dataset.ipynb : Contains the unique bagging model trained on dataset 2 [Note: Used MRMR as feature selection ]
5. model_testing.ipynb : Contains the metrics of the classifier models, without FS, that are used in the bagging classifier. 
6. pearson_selection_resampling.ipynb : Contains the code of the new model [ New feature selection + sample resampling + new bagging model where the bags are based on the samples ]
7. FS_classif.ipynb : Contains the metrics of the classifier models, with FS, that are used in the bagging classifier.
8. audio_extractor.py : A Streamlit application which basically process the audio file, to extract the features, you upload. [Check it out here](https://audio-extractor.streamlit.app/)

## Few points to note:

1. Files 3 and 4 are for understanding the classifier model. Note that here the classification model includes feature resampling and mrmr as feature selection method. 
2. Currently the files 5,6,7 contains "pd_speech_features.csv" as the dataset.
3. To use "Parkinson Disease.csv" simply replace the dataset and replace 'id' with 'name', 'class' with 'status'.

```python
import numpy as np
import pandas as pd

df = pd.read_csv("Parkinsson disease.csv")

# Data Cleaning
df.drop('name', axis=1, inplace=True)

# Data preprocessing
X= df.drop('status', axis=1)
Y= df['status']
```

```python
import numpy as np
import pandas as pd

df = pd.read_csv('pd_speech_features.csv')

# Data cleaning
df = df.drop('id', axis=1)  # Remove the 'name' column

# Data preprocessing
X = df.drop('class', axis=1) 
Y = df['class']  
```
