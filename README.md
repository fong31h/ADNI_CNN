# ADNI_CNN
Spencer Fong's work on the ADNI Research Project with Dr Rhodes and Dr Hedges (BYU). Feel free to reach out at fongspe28@gmail.com if you are looking at this stuff and have questions.

Overview of Work

My primary goal while working on this project was to train a Convolutional Neural Network to identify MRI brain scans from the ADNI study database as one of three classes: AD (Alzheimer's Disease), MCI (Mildly Cognitively Impaired), and CN (Cognitively Normal).  To this end, my most important work is found in the python scripts training_loop.py and training_loop_separate.py. These two scripts contain the code for an entire training loop with a pretrained, frozen, resnet50 model. They are meant to be run with 256x256 grayscale images, containing three classes. The script will both load in the pretrained resnet50 from the PyTorch resources and modify it to the task. The scripts will also record a running log of the loss, and iteratively save the best model. The only difference between the two scripts is training_loop.py is meant to run from one csv file that is broken into training, validation, and testing, while training_loop_separate.py requires three different csv files that are already broken into training, validation, and testing. You'll probably need some knowledge of neural networks and Pytorch to understand the other parameters, but it's relatively basic stuff.

These python scripts can be used to lead into my secondary goal of this project: producing embedded representations of MRI scans (basically the image in vector form) which would be used in a Manifold Alignment machine learning model (created by Dr Rhodes and others on the project) to combine multiple domains of data in order to achieve better prediction results than any of the domains could individually. The idea is, if a model can be trained that achieves better-than-guessing accuracy on the MRI scans (so >33% for three classes), this model now contains some information about classifying Alzheimer's that can be represented in the vector form embeddings. These vector form embeddings are created as part of the neural network process, and can be accessed by removing the final layer of the neural network (before the embeddings are turned into probabilities) and saving the output directly. Unfortunately, I was never able to create a customizable Python script to perform this function, but there is a script titled get_embeddings.py that contains an example of how to do this. Fortunately this is pretty simple and Google or AI could help you as well.

In working to these two goals, I spent the majority of my time data cleaning and wrangling. Familiarizing myself with the structure of the MRI scan data was the most important step in being successful on this project. In order to train the convolutional neural networks, the data has to be fed in from a CSV file that contains the paths to the scans and the labels. Writing a CSV file like this required navigating the file structure. To this end, I have included the script matrix_script.py, which takes in a directory of downloaded ADNI scan data, and converts the original DICOM files into numpy matrices. It creates both a 3D numpy matrix and individual 2D matrices for each scan. We originally hoped to train a 3D CNN with the 3D data, but it didn't really work so we pivoted to using just the 2D matrices. If you download the data directly from the ADNI website, the structure will not work with matrix_script.py. However, the process to structuring the directory correctly is very simple and is found in the 'Process' text file. The script also writes two different CSV files (2D and 3D), which are compatible to be used in the training loop.

Outside of these main things, there is a large amount of other work that I did progressively solving problems that arose. I won't be able to go into detail on everything here, but hopefully the majority of these problems are now accounted for and solved within the Python scripts included here. That being said, I will try to give an overview into a few problems that I did not fully solve before leaving off, which is to say, basically, where the project is now. First off, sometime in the last couple months we discovered that the labels that I had been using did not capture visit-to-visit variation in the diagnosis (i.e. some patient may have switched from CN to MCI, but my labels were only going off the original diagnosis). Therefore, the labels pulled by my matrix_script.py are not the most accurate possible. In order to train the CNN with the correct labels (I will call them the "variable" labels, since they can vary from visit to visit), I worked with Dallan Gardner who did work to pull together the variable labels. While this problem is mostly solved, any future data that I never worked with will require coordinating with Dallan again or using his work to get the variable labels.

Next, I will explain the reason for the existence of both training_loop.py and training_loop_separated.py. In my preliminary results and method, I was taking a big chunk of data and splitting it randomly into training, validation, and testing. However, this did not account for the fact that each subjects data would be split between all three groups. As we prepared to get real results from the CNN, I took every subject ID in the entire ADNI project and split them into training, validation, and testing, so that this cross-contamination would no longer happen. The directory 'Splitting_Data' contains the work I did to gather the subject IDs, and split them randomly into these three groups. Another Python script, split_dfs.py, is used to actually take a group of scan data and split into the three groups along the subject IDs. Unfortunately split_dfs.py is not super optimized either, but I'll try to explain the functionality below. So training_loop_separated.py exists because I needed to adapt the training loop to not take in one chunk of data and then split, but to take in three different chunks of data already split into training, validation, and testing. Now here's the problem though: unfortunately, removing the cross-contamination from subjects across the groups also hurt the accuracy quite a bit. I went from averaging around 70-80% accuracy, to 40-50% accuracy. Now if that accuracy is real, then that should be good enough for the purposes of this project, but it could be worth exploring any other tweaks that can be made to improve that accuracy.

So that leaves us basically where I left the project. In order to get real results, we decided to use the entirety of ADNI 1 MRI scans as our data set. So I've included several CSV files that will probably be helpful in using the ADNI 1 data. The path structure under the 'File Path' variable is the same one that would be created by following the 'Process' text file. The ADNI 1 data is technically all ready to go, so maybe you can get results from it. The data itself is about 100 GB and currently stored under my account on the stats servers. Dr Rhodes knows about this and should have access. Feel free to take my work and modify it however you want to fit what you want to do. Hopefully all this can in some way be helpful in minimizing the amount of work that gets repeated.

Full list of files

**Import_classes.py**
This file is contains several classes are called on at various points in the training loop scripts. All the classes are necessary in the running the scripts, but the most important one is Rage_Scans, which is the PyTorch dataset class that reads the CSV file containing paths and labels into the CNN for training.

**modify_resnet50.py**
This file contains the exact same code found in the training loop scripts to modify the Resnet50 model for this specific task.

**matrix_script.py**
This is the aforementioned script that will create numpy matrices and write ready-to-go CSV files. Unfortunately, the labels that this script pulls are not the most optimal labels, but they are pretty close to the variable ones and can be used for testing things before getting results with the correct ones. I would recommend using the paths that this script pulls and combining with variable labels that you get from Dallan Gardner's work.

**get_embeddings.py**

**examining_Dallans_data.ipynb**
After getting the variable labels from Dallan, they weren't in the correct order to be used with my CSV files, so I had to figure out how to reorder them and create a new CSV file. This is that work.

**Splitting_Data**
This is a directory that contains the work I did to split the subject IDs into training, validation, and testing. This is probably pretty confusing to look at since I did a lot of random stuff, but there shouldn't be any more work to do here. Data_split.csv contains the full list of Subject IDs and which group they belong to, this csv file is used in split_dfs.py.

**split_dfs.py**
As mentioned, split_dfs.py is used to take a group of scan data and split it into training, validation, and testing according to the Data_split.csv list. This script is set up to use a CSV file containing file paths and labels and then search these file paths using regular expressions for the subject ID. Then it compares that ID to Data_split.csv and places the path into either the training, validation, or testing data frame. Then it saves those data frames into CSV files that can be used with training_loop_separate.py.

**training_loop.py**
**training_loop_separate.py**

**2D_Images.zip**
This is the entire list of file paths and (non-variable) labels for the ADNI 1 dataset, ready to be used in training.

**2D_Images_variable.zip**
This is the same as 2D_Images.csv but with the variable labels. This was created in the examining_dallans_data.ipynb notebook. This one should be used for getting real results from training. It's much too long to examine manually, but if you compare the labels between 2D_Images.csv and 2D_Images_variable.csv, you'll see that only a small percentage of them have changed.

**3D_Images.csv**
This is the list of 3D file paths and non-variable labels for the ADNI 1 dataset. This one is just useful because there is one file path per complete MRI scan, rather than how the 2D scans have over 150 file paths for each visit (since the MRI scan is broken up into each 2D component).

**Models**
This is a directory containing some models I worked on. Unfortunately these were all trained with cross-contaminated data, so the results are probably artificially high. As I mentioned, I didn't save any of the non-cross-contaminated models since they kind of sucked. If you read 'Model Information,' it will reference 'Testing big download' and 'Big Attempt' these were small and medium sized data sets I was using throughout as example data sets to practice and learn with. You'll probably see them referenced elsewhere.

**Pixel_Importance**
**Pixel_Importance_Work.ipynb**
'Pixel_Importance' is a directory that contains the (ultimately unsuccessful) work I did to attempt to analyze what aspects of the images were helpful to the CNN in making predictions. Most of the process is found Pixel_Importance_Work.ipynb.

**Dim_red.ipynb**
This notebook file has some work I did to visualize the data using dimensionality reduction. The process of dimensionality reduction actually involves getting the vector embeddings I talked about earlier, so you could look through here for more help on doing that. I also have some work in here I did examining which slices of the MRI scans are more useful in predicting and which aren't. The result was that the edge slices are not important, but we haven't implemented that yet because we're not sure if the neuroscience people would approve of removing some sections of the brain scan.

**Diagnoses.ipynb**
This contains some work I did to see if the non-variable labels are the same as the DX_bl variable that Marshall Nielsen (other guy on the project) uses in his end of the project. Result is that yeah, they're pretty much the same.

**working on 3d matrix script.ipynb**
Just some of my work in creating matrix_script.py early on. Just included in case it is helpful in some way.

**Process**
Contains the general overview of my process from download to training. Includes the specifics of structuring the directories to match how I wrote the scripts.
