import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Class names from documentation
class_names = [ "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

#dsplay the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("CNN Confusion Matrix - Fashion MNIST")
plt.show()

misclassified_indices = np.where(y_pred != y_test)[0]

#3 misclassified images
plt.figure(figsize=(12, 4))

for i in range(3):
    idx = misclassified_indices[i]
    plt.subplot(1, 3, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(
        f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

# One pattern observed in the misclassifications, The model often confuses visually similar clothing categories, such as
#T-shirt/top, Shirt, Pullover, and Coat, because these classes can share similar shapes and outlines in small grayscale images.

#A realistic improvement would be to make the CNN deeper or add another
#convolution layer so the model can learn more detailed visual features. Data augmentation could also help the model generalize better.