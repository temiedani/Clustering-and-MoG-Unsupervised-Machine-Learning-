import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:, 0] = f1
X_full[:, 1] = f2
# /
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3
#########################################

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
'''
X_phoneme_1 = np.zeros((np.sum(phoneme_id == 1), 2))
X_phoneme_1 = X_full[phoneme_id == 1, :]
X_phoneme_2 = np.zeros((np.sum(phoneme_id == 2), 2))
X_phoneme_2 = X_full[phoneme_id == 2, :]
X_phonemes_1_2 = np.append(X_phoneme_1, X_phoneme_2, axis=0)
'''
true_id = []
X_phonemes_1_2 = []
for index in range(len(phoneme_id)):
    if phoneme_id[index] == 1 or phoneme_id[index] == 2:
        true_id.append(phoneme_id[index])
        X_phonemes_1_2.append(X_full[index])
X_phonemes_1_2 = np.array(X_phonemes_1_2)
# #######################################/
# Plot array containing the chosen phonemes
# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(
    os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)

#########################################
# load trained GMM data and get ,mu.s p for each model trained on phoneme 1.
GMM_1 = np.load("data/GMM_params_phoneme_01_k_03.npy", allow_pickle=True)
GMM_1 = np.ndarray.tolist(GMM_1)
# Extract Saved GMM parameters for phoneme 1.
mu_1 = GMM_1['mu']
p_1 = GMM_1['p']
s_1 = GMM_1['s']

# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
pred1 = get_predictions(mu_1, s_1, p_1, X_phonemes_1_2)
# pred1 = normalize(pred1, axis=1, norm='l1')

# likelihood is the sum of the probablity of the k GMM components
likelihood_1 = np.sum(pred1, axis=1)


# load trained GMM data and get ,mu.s p for each model trained on phoneme 2.
GMM_2 = np.load("data/GMM_params_phoneme_02_k_03.npy", allow_pickle=True)
GMM_2 = np.ndarray.tolist(GMM_2)

# Extract Saved GMM parameters for phoneme 2.
mu_2 = GMM_2['mu']
p_2 = GMM_2['p']
s_2 = GMM_2['s']

# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
pred2 = get_predictions(mu_2, s_2, p_2, X_phonemes_1_2)
# pred2 = normalize(pred2, axis=1, norm='l1')

# likelihood is the sum of the probablity of the k GMM components
likelihood_2 = np.sum(pred2, axis=1)

# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
predicted_id = np.zeros(phoneme_id.shape)
for index in range(len(pred1)):
    if likelihood_1[index] > likelihood_2[index]:
        predicted_id[index] = 1.0
    else:
        predicted_id[index] = 2.0
# /
accuracy = 0.0
counter = 0
for index in range(X_phonemes_1_2.shape[0]):
    if predicted_id[index] == true_id[index]:
        counter += 1
accuracy = 100 * (counter/X_phonemes_1_2.shape[0])

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))
################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
