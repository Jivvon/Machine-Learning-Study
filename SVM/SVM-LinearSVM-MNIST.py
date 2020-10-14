import numpy as np
import gzip

### 데이터 가져오기
"""
http://yann.lecun.com/exdb/mnist/ 에 다음과 같이 설명되어 있다.

```
TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
```

데이터가 다음과 같이 시작하므로 
magic number, number of images, rows, columns = 총 16바이트를 읽은 후 픽셀값을 읽기 시작한다.
number of rows와 columns가 각각 28,28이므로 pixels = image_size * image_size 를 기준으로 나누어 보면 될 것 같다.
즉, 데이터는 (이미지 number, row, column, pixel)로 나타낼 수 있을 것이다.
"""

image_size = 28 # 이미지 크기 28x28
num_images = 1000 # 이미지의 개수
pixels = image_size * image_size

raw_data = []
with gzip.open('train-images-idx3-ubyte.gz', 'r') as f:
    f.read(16)
    for line in f:
        this_line = np.frombuffer(line, dtype=np.uint8).astype(np.float32)
        raw_data.extend(this_line)
raw_data = np.array(raw_data) # (47040000,)
data = raw_data.reshape(-1, pixels) # (60000, 784) -> 이미지 60000개

"""
각 이미지는 28 x 28로 설명되어 있으니 pixels = image_size * image_size로 지정한다.
이미지별로 데이터를 구분하기 위해 (-1,pixels)로 reshape하면 총 60000개의 이미지가 있는 것 을 알 수 있다. 
"""

# 첫 번째 데이터를 출력하면 다음과 같다.
# 784개의 pixels 데이터를 (가로, 세로, 값)으로 reshape하여 출력해보았다.
import matplotlib.pyplot as plt
tmp_data = data[0].reshape(image_size,image_size,1)
image = np.asarray(tmp_data).squeeze()
plt.imshow(image)
plt.show()

data = data[:num_images] # 필요한 만큼만 남겨두자
data.shape # 5000(, 784)


### 데이터 라벨(정답) 가져오기
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
labels = []
for i in range(0,num_images):   
    buf = f.read(1)
    label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    labels.append(*label)
labels = np.array(labels)


### train 데이터와 test 데이터로 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33, random_state = 1)

# train_test_split을 이용하여 훈련용 데이터와 테스트 데이터로 나눈다.

print(X_train[15,546], max(X_train[15])) # 253.0 255.0
# 이미지 데이터이기 때문에 픽셀 범위의 데이터 이다. (0 ~ 255)
# 이를 0~1 범위로 정규화 시켜준다.

X_train = X_train / 255
X_test = X_test / 255
print(X_train[15,546], max(X_train[15])) # 0.99215686 1.0
# 0~1 범위의 값으로 정규화가 되었음을 알 수 있다.

# 이제 Linear SVM 모델을 통해 학습시켜보자.
from sklearn.svm import LinearSVC, SVC
linearSVC_model = LinearSVC(C=1, random_state=1, max_iter=100000)
clf = linearSVC_model.fit(X_train, y_train) # 훈련 데이터로 학습시키고
y_predict = clf.predict(X_test) # 테스트 데이터로 예측을 해본다.

# 테스트 데이터의 예측된 값(y_predict)이 정답(y_test)을 맞춘 비율
len(y_predict[y_predict == y_test]) / len(y_test)


# 이제 nonlinear SVM 모델을 통해 학습시켜보자.
SVC_model = SVC(kernel='rbf', C=1, gamma=0.1, max_iter=100000)
clf = SVC_model.fit(X_train, y_train) # 훈련 데이터로 학습시키고
y_predict = clf.predict(X_test) # 테스트 데이터로 예측을 해본다.

# 테스트 데이터의 예측된 값(y_predict)이 정답(y_test)을 맞춘 비율
len(y_predict[y_predict == y_test]) / len(y_test)

# 보다 정확한 성능 측정을 위해 Cross Validation을 사용해보면 다음과 같다.
from sklearn.model_selection import cross_val_score
linear_SVM = np.mean(cross_val_score(linearSVC_model, X_train, y_train, scoring=None, cv=None))
SVM = np.mean(cross_val_score(SVC_model, X_train, y_train, scoring=None, cv=None))

print(f'LinearSVM : {linear_SVM}\nSVM : {SVM}')
# LinearSVM : 0.8725373134328359
# SVM : 0.817910447761194
