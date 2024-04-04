import os
import cv2
import numpy as np
def rename_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            name, ext = os.path.splitext(filename)
            parts = name.split(" - ")
            if len(parts) > 1:
                new_name = parts[0].replace(" ", "X") + ext
                new_filename = os.path.join(directory, new_name)
                counter = 1
                while os.path.exists(new_filename):
                    new_name = parts[0].replace(" ", "") + "_" + str(counter) + ext
                    new_filename = os.path.join(directory, new_name)
                    counter += 1
                os.rename(os.path.join(directory, filename), new_filename)
                print(f"Renamed {filename} to {new_name}")

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                images.append(img)
    return images

def preprocess_images(images):
    # Write a Preprocessing Method
    return images

def extract_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)
        features.append(circles)
    return features

def train_model(features, labels):
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_LINEAR)
    model.train(np.array(features), cv2.ml.ROW_SAMPLE, np.array(labels))
    return model

def main():
    directory = r"C:/Users/prajw/Python/Mech_Project/images/200 X 50 X 20/with_dimensions"
    rename_images(directory)
    """with_dimensions_dir = "C:/Users/prajw/Python/Mech_Project/images/200 X 50 X 0/with_dimensions"
    without_dimensions_dir = "C:/Users/prajw/Python/Mech_Project/images/200 X 50 X 0/without_dimensions"
    with_dimensions_images = load_images(with_dimensions_dir)
    without_dimensions_images = load_images(without_dimensions_dir)
    with_dimensions_images = preprocess_images(with_dimensions_images)
    without_dimensions_images = preprocess_images(without_dimensions_images)
    with_dimensions_features = extract_features(with_dimensions_images)
    without_dimensions_features = extract_features(without_dimensions_images)
    with_dimensions_labels = label_images(with_dimensions_images)
    without_dimensions_labels = label_images(without_dimensions_images)

    # Combine features and labels for training
    features = with_dimensions_features + without_dimensions_features
    labels = np.concatenate([with_dimensions_labels, without_dimensions_labels])
    model = train_model(features, labels)

    # Save the trained model
    model.save("circle_detection_model.xml")"""

if __name__ == "__main__":
    main()
