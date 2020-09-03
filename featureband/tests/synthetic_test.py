from sklearn.datasets import make_classification

from featureband.feature_band import FeatureBand
from featureband.util.metrics_util import load_clf

FINAL_CLASSIFIER = "knn"  # ["knn", "logistic", "decision_tree"]
selected_num = 10

x, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                           n_redundant=10, n_classes=2, random_state=42)
clf = load_clf(FINAL_CLASSIFIER)

fb = FeatureBand(r0=80, n0=1000, clf=clf, k=selected_num, population_size=5, max_iter=20)
fb.fit(x, y, "accuracy")
fb.transform(x)

print("clf:", FINAL_CLASSIFIER, "selected_num:", selected_num)
