from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np


# KmeansMoons와 KmeansBlobs의 부모 클래스
class KmeansBase():
    def __init__(self):
        pass
    
    def createFigures(self, row=3, col=3):
        fig = plt.figure(figsize=(20,20))
        axs = fig.subplots(row, col, sharex=False, sharey=False)
        axs = axs.reshape(1,-1)[0]
        return fig, axs
    
    def draw_plot(self, ax, X, y, n_cluster):
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(X)
        y_pred = kmeans.predict(X)
        ax.scatter(X[:,0], X[:,1], s=60, c=y_pred, marker='o', edgecolors='k')
        return y_pred

    def run(self):
        pass


# make_moons 데이터를 이용하여 Kmeans 수행
class KmeansMoons(KmeansBase):
    def __init__(self):
        super(KmeansMoons, self).__init__()

    def createData(self) ->(np.ndarray, np.ndarray):
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)
        return (X, y)

    def run(self):
        X, y = self.createData()
        fig, axs = self.createFigures()
        for i, ax in enumerate(axs):
            self.draw_plot(ax, X, y, n_cluster=i+2)
            ax.set_title(f'n_cluter : {i+2}')
        plt.show()


# make_blobs 데이터를 이용하여 Kmeans 수행
class KmeansBlobs(KmeansBase):
    def __init__(self):
        super(KmeansBlobs, self).__init__()

    def createData(self, cluster_std) ->(np.ndarray, np.ndarray):
        from sklearn.datasets import make_blobs
        if type(cluster_std) == list:
            centers = len(cluster_std)
        else:
            centers = 1
        X, y = make_blobs(n_samples = 200, centers = centers, cluster_std=cluster_std, random_state = 0)
        return (X, y)
    
    def createFigure(self):
        return plt.figure(figsize=(5,5))
        
    def run(self, cluster_std:list=[1.0], n_cluster=2):
        if len(cluster_std) == 1:
            cluster_std = cluster_std.pop()
        X, y = self.createData(cluster_std)
        fig = self.createFigure()
        y_pred = self.draw_plot(plt, X, y, n_cluster=n_cluster)
        plt.title(f'cluster_std : {cluster_std} n_cluter : {n_cluster}')
        plt.show()


if __name__ == '__main__':
    km = KmeansMoons()
    km.run()

    kb = KmeansBlobs()
    kb.run([1.0])

    kb.run([1.0, 2.0, 1.5], 2)
    kb.run([1.0, 2.0, 1.5], 3)
    kb.run([1.0, 2.0, 1.5], 4)

    kb.run([5, 5, 5, 5, 5], 2)
    kb.run([5, 5, 5, 5, 5], 3)
    kb.run([5, 5, 5, 5, 5], 5)

    kb.run([1, 3, 5, 4, 5])

    kb.run([.1, .3, .5, .4, .5], 3)
    kb.run([.1, .3, .5, .4, .5], 4)
    kb.run([.1, .3, .5, .4, .5], 5)
