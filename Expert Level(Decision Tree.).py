import cv2
import sklearn
import glob

from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
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
max_depth = [1, 2, 4, 8, 16, 32, 64]
accuracy_list = []
for d in max_depth:
    # 创建一个决策树分类器
    dtc = DecisionTreeClassifier(max_depth=d)
    # 训练分类器
    dtc.fit(X_train, y_train)
    # 预测
    y_pred = dtc.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    print(f"max_depth = {d} D Accuracy:{accuracy}")

plt.clf()
plt.plot(max_depth, accuracy_list, '-or')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.title('Decision Tree Accuracy, max_depth vs accuracy')
plt.show()


