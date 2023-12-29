#ROC CURVE

# Create a dictionary to map class labels to class names
class_names = {
    0: 'glioma',
    1: 'meningioma',
    2: 'notumor',
    3: 'pituitary'
}


# Binarize the true and predicted labels
lb = LabelBinarizer()
true_labels_bin = lb.fit_transform(true_labels)

# Create an array to store the predicted labels in a binarized form
predicted_labels_bin = lb.transform(predicted_labels)

fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = num_classes  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predicted_labels_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 6))

colors = ['b', 'g', 'r', 'c']  # You may need to extend the list for more classes

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve ({class_names[i]}) (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')

# Modify legend labels to use class names
legend_labels = [f'{class_names[i]} (AUC = {roc_auc[i]:.2f})' for i in range(n_classes)]
plt.legend(legend_labels, loc="lower right")
plt.show()

# You can also compute the macro-average AUC, which is an average AUC over all classes
macro_roc_auc = sum(roc_auc.values()) / n_classes
print(f"Macro-average AUC: {macro_roc_auc:.2f}")

