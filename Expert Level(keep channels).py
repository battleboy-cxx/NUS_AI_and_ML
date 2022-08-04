import cv2
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here
dataset_path = "E:\\NUS\\images\\"

X = []
y = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)

    # write code to read ecah file i
    img = cv2.imread(i)
    # and append it to list X
    X.append(img)

# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________


# T2 start __________________________________________________________________________________
# Preprocessing

X_processed = []
for x in X:
    # Write code to resize image x to 48x48 and store in temp_x
    temp_x = cv2.resize(x, (48, 48))
    # Keep 3 channels (RGB)
    temp_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2RGB)
    # Append the converted image into X_processed
    X_processed.append(temp_x)

# T2 end ____________________________________________________________________________________


# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:

    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)
    X_features.append(x_feature)

# write code to Split training & testing sets using sklearn.model_selection.train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

#T3 end ____________________________________________________________________________________



#T4 start __________________________________________________________________________________
# train model
clf = SVC(kernel='rbf', C=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
print("Keep Three Channels Accuracy:", accuracy)
