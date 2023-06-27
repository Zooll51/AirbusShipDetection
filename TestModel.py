import os
from Model import Create_Model
import pandas as pd
from OtherFile import create_test_generator

###################################################################################################
# Create and save checkpoint

batch_size = 10
checkpoint_path = "working/my_checkpoint.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

###################################################################################################
# Create model and apply weights

model = Create_Model()
model.load_weights(checkpoint_path)

###################################################################################################

# Read data
sub_df = pd.read_csv("sample_submission_v2.csv")
# Prediction
test_generator = create_test_generator(batch_size, sub_df)