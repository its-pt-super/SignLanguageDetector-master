# Import necessary libraries
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    Plot a confusion matrix using matplotlib.

    Parameters:
    - cm: confusion matrix from sklearn.metrics.confusion_matrix
    - target_names: list of classification classes (e.g., ['high', 'medium', 'low'])
    - title: text to display at the top of the matrix
    - cmap: color map for the matrix
    - normalize: If True, plot the proportions; If False, plot the raw numbers

    Usage:
    plot_confusion_matrix(cm=cm, normalize=True, target_names=y_labels_vals, title=best_estimator_name)
    """
    import itertools

    # Calculate accuracy and misclassification rate
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    # Set default colormap if not specified
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # Create a figure for the plot
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Set ticks and labels on axes
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    # Normalize confusion matrix if required
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Determine threshold for text color
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    # Display values in each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # Adjust layout and labels
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    # Save the plot as an image
    plt.savefig('confusion_matrix.png')


# Load the pre-trained Keras model
model = load_model('model/keypoint_classifier/keypoint_classifier_new.h5')

# Initialize empty lists for predicted labels and true labels
pred_labels = []

# Measure the time taken for predictions
start_time = time.time()
pred_probabs = model.predict(X_test)
end_time = time.time()
pred_time = end_time - start_time
avg_pred_time = pred_time / X_test.shape[0]
print('Average prediction time: %fs' % (avg_pred_time))

# Convert predicted probabilities to labels
for pred_probab in pred_probabs:
    pred_labels.append(list(pred_probab).index(max(pred_probab)))

# Calculate confusion matrix
cm = confusion_matrix(y_test, np.array(pred_labels))

# Generate and print classification report
classification_report_str = classification_report(y_test, np.array(pred_labels))
print('\n\nClassification Report')
print('---------------------------')
print(classification_report_str)

# Plot and save the confusion matrix
plot_confusion_matrix(cm, range(44), normalize=False)



