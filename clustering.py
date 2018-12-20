import json
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_dataset():
    texts = []
    labels = []
    with open(r"D:\python project\data\Tweets.txt") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["cluster"])
    return texts, labels

def vectorize(texts):
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, max_features=1000, use_idf=True,stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    return vectors.todense()

def evaluate(labels, prediction):
    NMI = metrics.normalized_mutual_info_score(labels, prediction)
    print("NMI:", "%.4f" % NMI)

# K_Means
def kmeans(X, Y, random_state):
    print('k-means', end="\t")
    K = len(set(Y))
    prediction = KMeans(n_clusters=K, random_state=random_state).fit_predict(X)
    evaluate(Y, prediction)

# Affinity propagation
def affinityPropagation(X, Y):
    print('AffinityPropagation', end="\t")
    prediction = AffinityPropagation().fit_predict(X)
    evaluate(Y, prediction)

# Mean-Shift
def meanShift(X, Y):
    print('MeanShift', end="\t")
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    prediction = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(X)
    evaluate(Y, prediction)

# SpectralClustering
def spectralClustering(X, Y, gamma):
    print('spectral_clustering', end="\t")
    K = len(set(Y))
    prediction = SpectralClustering(n_clusters=K, gamma=gamma).fit_predict(X)
    evaluate(Y, prediction)

# AgglomerativeClustering
def agglomerativeClustering(X, Y, linkage):
    print('AgglomerativeClustering: ' + linkage, end="\t")
    K = len(set(Y))
    prediction = AgglomerativeClustering(n_clusters=K, linkage=linkage).fit_predict(X)
    evaluate(Y, prediction)

# DBSCAN
def dbscan(X, Y, eps, min_samples):
    print('DBSCAN', end="\t")
    prediction = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    evaluate(Y, prediction)

# GaussianMixture
def gaussianMixture(X, Y, cov_type):
    print('GaussianMixture: ' + cov_type, end="\t")
    K = len(set(Y))
    gmm = GaussianMixture(n_components=K, covariance_type=cov_type).fit(X)
    prediction = gmm.predict(X)
    evaluate(Y, prediction)

def main():
    texts, labels=generate_dataset()
    vectors=vectorize(texts)
    print("--------------------------------------------")
    kmeans(vectors, labels, random_state=13)
    print("--------------------------------------------")
    affinityPropagation(vectors, labels)
    print("--------------------------------------------")
    meanShift(vectors, labels)
    print("--------------------------------------------")
    spectralClustering(vectors, labels, gamma=0.06)
    print("--------------------------------------------")
    linkages = ['ward', 'average', 'complete']
    for linkage in linkages:
        agglomerativeClustering(vectors, labels, linkage)
    print("--------------------------------------------")
    dbscan(vectors, labels, eps=0.3, min_samples=1)
    cov_types = ['spherical', 'diag', 'tied', 'full']
    for cov_type in cov_types:
        gaussianMixture(vectors, labels, cov_type)

main()