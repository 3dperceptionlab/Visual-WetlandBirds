import numpy as np

# Add syspath
import sys
sys.path.append('modules')
from dataset import get_dataloader

get_dataloader(statistics=True)

activities = ['Feeding', 'Preening', 'Swimming', 'Walking', 'Alert', 'Flying', 'Resting']

id_to_label = {i: activities[i] for i in range(len(activities))}
label_to_id = {v: k for k, v in id_to_label.items()}

# Load numpy array from file
activity_ids = np.load('annotations/stats/activities.npy')

# Remove the 4s from the activity_ids np array, example [1,2,3,3,3,2,4,2] -> [1,2,3,3,3,2,2]

activity_frames = {}

# activity_ids = activity_ids[activity_ids != 4]

for i in range(len(activities)):
    # if i != label_to_id['None']:
    activity_frames[i] = np.load(f'annotations/stats/activity_{i}_lengths.npy')

total_frames = np.concatenate(list(activity_frames.values()))
print('Statistics:')
print('Total Frames:', len(total_frames))
print('Mean:', np.mean(total_frames))
print('Median:', np.median(total_frames))
print('Std Dev:', np.std(total_frames))

# Dictionary of the number of frames per activity
frames_per_activity = {k: len(v) for k, v in activity_frames.items()}

print(len(activity_ids))
print(frames_per_activity)
print(len(activity_frames))

# Build a graphic and store it in the stats folder, with the number of activities in number_activites. Another with the median, mean, std, min and max of the lengths of each activity in activity_lengths
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Count the occurrences of each activity
unique, counts = np.unique(activity_ids, return_counts=True)
activity_counts = dict(zip(unique, counts))

# Convert IDs to labels for plotting
labels = [id_to_label[id] for id in activity_counts.keys()]
counts = list(activity_counts.values())


# Store the plot in a PDF file
pdf_filename = "annotations/stats/activity_counts.pdf"

with PdfPages(pdf_filename) as pdf:
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Activity')
    plt.ylabel('Count')
    plt.title('Number of Activities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig()


# Convert IDs to labels for plotting
labels = [id_to_label[id] for id in frames_per_activity.keys()]
counts = list(frames_per_activity.values())


# Store the plot in a PDF file
pdf_filename = "annotations/stats/frames_per_activity_counts.pdf"

with PdfPages(pdf_filename) as pdf:
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Activity')
    plt.ylabel('Count')
    plt.title('Number of Frames per Activities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig()


# Calculate statistics for each activity
means = {id_to_label[k]: np.mean(v) for k, v in activity_frames.items()}
medians = {id_to_label[k]: np.median(v) for k, v in activity_frames.items()}
std_devs = {id_to_label[k]: np.std(v) for k, v in activity_frames.items()}

# Prepare data for plotting
labels = list(means.keys())
mean_values = list(means.values())
median_values = list(medians.values())
std_dev_values = list(std_devs.values())

# Plotting with median included
pdf_filename_median = "annotations/stats/activity_length_statistics_with_median.pdf"
with PdfPages(pdf_filename_median) as pdf:
    plt.figure(figsize=(10, 6))

    # Mean, Median, and Standard Deviation Bar Plot
    x = np.arange(len(labels))
    width = 0.2

    plt.bar(x - width, mean_values, width, label='Mean', color='skyblue')
    plt.bar(x, median_values, width, label='Median', color='salmon')
    plt.bar(x + width, std_dev_values, width, label='Std Dev', color='lightgreen')

    plt.xlabel('Activity')
    plt.ylabel('Frames')
    plt.title('Mean, Median, and Standard Deviation of Frames per Activity')
    plt.xticks(x, labels, rotation=45)
    plt.legend()

    plt.tight_layout()
    pdf.savefig()