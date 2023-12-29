#Confusion Matrix

# Initialize lists to store predictions and ground truth labels
all_preds = []
all_labels = []
class_names = ["Glioma", "Menin", "Notu", "Pitu"]
# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to NumPy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate confusion matrix
confusion_mat = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(4, 4),dpi=300)
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()