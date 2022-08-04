import cv2
import sklearn
import glob

from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
    # add a gaussian blur to reduce noise
    temp_x = cv2.GaussianBlur(temp_x, (5, 5), 0)
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
# random forest

# 创建knn分类器
n_list = [1,3,5,7,9]
accuracy_list = []
for i in n_list:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    # 训练分类器
    knn.fit(X_train, y_train)
    # 预测
    y_pred = knn.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    print(f"KNN {i} Neighbors Accuracy: {accuracy}")

plt.clf()
plt.plot(n_list, accuracy_list)
plt.xlabel('n_neighbors')
plt.ylabel('accuracy')
plt.title('KNN Accuracy, n_neighbors vs accuracy')
plt.show()




