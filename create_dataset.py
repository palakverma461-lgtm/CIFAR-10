import os
import numpy as np
import tensorflow as tf
from PIL import Image

print("CIFAR-10 dataset se images extract karke folders bana rahe hain...")
(X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Base dataset folder create karein
os.makedirs("dataset", exist_ok=True)

for i, class_name in enumerate(class_names):
    class_dir = os.path.join("dataset", class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Is class ki images ke indices nikalein
    indices = np.where(y_train.flatten() == i)[0]
    
    # 10 random images select karein
    selected_indices = np.random.choice(indices, 10, replace=False)
    
    for idx, img_idx in enumerate(selected_indices):
        img_data = X_train[img_idx]
        img = Image.fromarray(img_data)
        img.save(os.path.join(class_dir, f"img_{idx+1}.jpg"))

print("Kaam ho gaya! 'dataset' folder ke andar saari 10-10 images automatically save ho gayi hain.")