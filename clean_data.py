# clean_data.py
# This file cleans the data collected from the simulation: training mode 
# After the data has been purged it can be used by model.py for training
# by Bryan Laygond

# USAGE:
#python clean_data.py --input driving_log.csv --output custom.csv


import argparse
import numpy as np
import pandas as pd

# construct argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
    help = "path to the input log")
ap.add_argument("-o", "--output", required = True,
    help = "path to output directory to store output log")
args = vars(ap.parse_args())

# Load input log and create dataframe
# Since our dataset does not include a header, set header=None and define header names
header_names =[
    'center', 'left', 'right',   # Directory paths to Camera Images 
    'steering',                  # steering angle between [-1,1]
    'throttle',                  # Trottle between [0,1]
    'break',                     # Break or Reverse Throttle [0,1] 
    'speed'                      # Speeed in mph
]
df = pd.read_csv(args["input"], header=None, names=header_names)

# I always ended up crashing the car at the end of the simulation
# Remove last 5 seconds of data
fps = 30        #data points per second
df = df.head(-fps*5)

# # I used to steer left and right before starting
# # Keep only data with forward speed higher than 5mph to purge starting of engine
# df = df.loc[(df["throttle"] > 0) & (df["speed"] > 5)]

# Shuffle data
df = df.sample(frac=1)      #fraction of rows to return in random order

# Even out steering angle data to prevent model from being bias
# Create histogram of steering angles and clip data
custom = pd.DataFrame() 	# custom balanced dataset
bins   = 1000				# Number of bins
clip   = 200                # Max number of steering angle data points per bin 
start  = 0                  # starting bin width range
for end in np.linspace(0, 1, num=bins):
    # Filter, Clip, and Add  
    df_bin = df.loc[(np.absolute(df["steering"]) >= start) & (np.absolute(df["steering"]) < end)]
    df_bin = df_bin.head( min(clip, df_bin.shape[0]) )
    custom = pd.concat([custom, df_bin])
    start = end

# Save output as csv
custom.to_csv(args["output"], index=False, header=False)


# #%%
# # Comparison between original and custom dataframes
# import pandas as pd

# # Since our dataset does not include a header, set header=None and define header names
# header_names =[
#     'center', 'left', 'right',   # Camera Images directory path
#     'steering',                  # steering angle between [-1,1]
#     'throttle',                  # Trottle between [0,1]
#     'break',                     # Break or Reverse Throttle [0,1] 
#     'speed'                      # Speeed in mph
# ]
# path1 = "../MYDATA/driving_log.csv"
# path2 = "../MYDATA/custom.csv"

# df = pd.read_csv(path1, header=None, names=header_names)
# custom = pd.read_csv(path2, header=None, names=header_names)

# # Create a temporal dataframe to make all steering positive and plot histogram
# bins=1000
# tmp = df 
# tmp.loc[tmp["steering"] < 0 ,  "steering"] *=-1
# tmp["steering"].hist(bins=bins)

# tmp = custom 
# tmp.loc[tmp["steering"] < 0 ,  "steering"] *=-1
# tmp["steering"].hist(bins=bins)