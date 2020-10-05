from matplotlib import pyplot as plt
import numpy as np

class KNN:
    def __init__(self, n_neighbors:int=1):
        """
        Args:
            n_neighbors (int, optional): k of k-nearest neighbors. Defaults to 1.
        """
        self.k = n_neighbors
    
    def set_data(self, data:list):
        """
        Args:
            data (list): [x_train, x_test, y_train, y_test]
        """
        (self.x_train, self.x_test, self.y_train, self.y_test) = data

    def distance(self, p1: list, p2: list) -> float:
        """get a distance between p1 and p2

        Args:
            p1 (list): point 1
            p2 (list): point 2

        Returns:
            float: distance between p1 and p2
        """
        return np.sqrt(np.sum(np.power((p2-p1),2)))

    # 가까운 순으로 인덱스를 리턴
    def get_sorted_neighbors(self, point: list) -> np.ndarray:
        """sort neighbors around point

        Args:
            point (list): a point which is gotten a distance between all neighbors

        Returns:
            np.ndarray: numpy array which contains sorted index of neighbors by distance with the point(args)
        """
        distances = list()
        for this_x_train in self.x_train: 
            distances.append(self.distance(this_x_train, point))
        return np.array(sorted(list(enumerate(distances)), key=lambda x:x[1]), dtype=np.uint)[:,0]

    def get_y(self, x_indexs:list, train_or_test:str='train') -> float:
        """calculate a most common class(y) matched with x_indexs(args)]

        Args:
            x_indexs (list): indexs of k neighbors
            train_or_test (str, optional): where you want to seek in between 'train' or 'test'. Defaults to 'train'.

        Returns:
            float: the most common value of y matched with x_indexs
        """
        from collections import Counter
        if train_or_test == 'train':
            return Counter(np.ravel(self.y_train[x_indexs])).most_common(n=1)[0][0] # 상위 1개의 최빈값
        elif train_or_test == 'test':
            return Counter(np.ravel(self.y_test[x_indexs])).most_common(n=1)[0][0] # 상위 1개의 최빈값

    # 정답 비율 리턴
    def validate(self) -> float:
        """calculate a ratio of correct answers from current train_data and test_data

        Returns:
            float: a ratio of correct answers from current train_data and test_data
        """
        nearlists = []
        answer_count = 0
        for this_x_test, this_y_test in zip(self.x_test, self.y_test):
            nearlists = self.get_sorted_neighbors(this_x_test)[:self.k] # 가장 가까운 k 개의 점들 구하고
            predict_y = self.get_y(nearlists,'train') # 정답을 구한다
            if predict_y == this_y_test: # 정답이 실제 y과 같다면 answer_count + 1
                answer_count += 1
        return answer_count / len(self.x_test)

    def run(self, *args) -> float:
        """excute the KNN.

        Args:
            *args : dataset would be given 

        Returns:
            float: return value of validate()
        """
        if args:
            self.set_data(args)
        ratio = self.validate()
        return ratio
            

class CrossValidation:
    def __init__(self, data:np.ndarray, num: int):
        self.data = np.random.shuffle(data) # 데이터 섞기
        self.dataset = np.split(data, num) # list of ndarray
    
    def get_train_test(self, nth:int):
        """
        :param nth: what number do you use to test data
        :result: list of [x_train, x_test, y_train, y_test]
        """
        
        train_data = np.delete(self.dataset, nth, axis=0).reshape(-1,len(self.dataset[0][0]))
        test_data = self.dataset[nth]
        x_train = train_data[:,:-1]
        x_test = test_data[:,:-1]
        y_train = train_data[:,-1].reshape(-1,1)
        y_test = test_data[:,-1].reshape(-1,1)

        return [x_train, x_test, y_train, y_test]
        

if __name__ == '__main__':
    data = np.loadtxt('./iris.csv', delimiter=",", dtype=np.float32)
    k, cross_val = 1, 5
    knn = KNN(k)
    crossValidation = CrossValidation(data, cross_val)
    total_ratio = 0
    for i in range(cross_val):
        this_data = crossValidation.get_train_test(i)
        knn.set_data(this_data)
        ratio = knn.run()
        total_ratio += ratio
        print('{} : {:.0f}%'.format(i, ratio * 100))
    print('평균 : {:.0f}%'.format(total_ratio / cross_val * 100))
