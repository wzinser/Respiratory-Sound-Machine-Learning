# Classifying Respiratory Sounds with Machine Learning

The purpose of this is to use automated machine learning algorithms to classify audio samples to aid in the detection of respiratory conditions (wheezes/crackles). 

## Getting Started

The database that contains all of the respiratory sounds and labels can be located at: https://bhichallenge.med.auth.gr/

### Prerequisites

Python 3.3
 - TensorFlow Library
 - Numpy Library

All code for this project was written and executed in Jupyter Notebook on Ubuntu 18.04 LTS (Virtual Machine). Python packages were managed with Anaconda. 

### Data Labels

Each entry in the database consists of an audio file (.wav) and a text file that labels each respiratory cycle in the following format

```
Example:
Inhale Exhale Crackles Wheeze
1.05    3.60     0       0
```

The entire databse is 1.8GB and can be downloaded from the following link: https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip

The following script can be ran to split .TXT label file into 4 separate .CSV files (start time, end time, wheezes, crackles) using the code below.  The start and end times are not required for the neural network and can be commented out if not needed.

```
import os
import pandas as pd
import numpy as np

directory_in_str = "/home/billy/Desktop/Python02"
directory = os.fsencode(directory_in_str)

#Create empty vectors to hold labels
time_start = []
time_end = []
crackles_labels = []
wheeze_labels = []

#Iterate over entire directory in folder
for file in sorted(os.listdir(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
             fo = open(filename)
             lines = [line.split() for line in fo]
             time_s =  [word[0] for word in lines]
             time_e = [word[1] for word in lines]
             crackles = ([word[2] for word in lines])
             wheezes = [word[3] for word in lines]
             
             #Each iteration add values to array   
             time_start.append(time_s)
             time_end.append(time_e)
             crackles_labels.append(crackles)
             wheeze_labels.append(wheezes)

#Create DataFrames                 
df_TimeStart = pd.DataFrame(time_start)             
df_TimeEnd   = pd.DataFrame(time_end)
df_Crackles  = pd.DataFrame(crackles_labels)    
df_Wheezes   = pd.DataFrame(wheeze_labels)


#Save DataFrames as .CSV file
df_TimeStart.to_csv('Time_Start.csv', index = False, header= False)
df_TimeEnd.to_csv('Time_End.csv', index = False, header= False)
df_Crackles.to_csv('Crackles Labels.csv', index = False, header= False)
df_Wheezes.to_csv('Wheezes Labels.csv', index = False, header= False)
```

Each text file is read and converted into a [1 x (# of breaths)] array. For example the following text file containing the labels would be converted to a csv file containing:

```
Example:
#  Inhale  Exhale Crackles Wheeze
1   0.021	  0.407	    0	      0 
2   0.407	  2.122	    0	      0
3   2.122	  3.522	    1	      0
4   3.522	  5.921	    0	      0
5   5.921	  7.793	    1	      0
6   7.793	  9.639	    0	      0
7   9.639	  11.291	   0	      0
8   11.291	 13.213	   1       0
9   13.213	 14.894	   1	      0
10  14.894	 16.383	   1	      0
11  16.383	 17.862	   0	      0
12  17.862	 19.073	   0	      0
13  19.073	 19.951	   0	      0

Crackle_Label = [0 0 1 0 1 0 0 1 1 1 0 0 0] 
Wheeze_Label  = [0 0 0 0 0 0 0 0 0 0 0 0 0]

```
The script iterates through each file in the folder, appends the array for each .txt file, adds all of them to a dataframe, and then converts the dataframe to a comma separated value file.


| #  | Inhale | Exhale | Crackle | Wheeze |
| ---|--------| ------ | --------|------- |
|1   |  0.021 | 0.407  |	    0   |	      0| 

|2   |0.407|	  2.122	|    0|	      0|
|3   |2.122|	  3.522	|    1|	      0|
|4   |3.522|	  5.921	|    0|	      0|
|5   |5.921|	  7.793	|    1|	      0|
|6   |7.793|	  9.639	|    0|	      0|
|7   |9.639|	  11.291|	   0|	      0|
|8   |11.291|	 13.213|	   1|       0|
|9   |13.213|	 14.894|	   1|	      0|
|10  |14.894|	 16.383|	   1|	      0|
|11  |16.383|	 17.862|	   0|	      0|
|12  |17.862|	 19.073|	   0|	      0|
|13  |19.073|	 19.951|	   0|	      0|
