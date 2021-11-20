#!python3

from sklearn import tree, ensemble, neighbors, naive_bayes, svm

models = {
    "decision_tree": tree.DecisionTreeClassifier(),
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "random_forest": ensemble.RandomForestClassifier(),
    "random_forest_5": ensemble.RandomForestClassifier(n_estimators=5),
    "random_forest_50": ensemble.RandomForestClassifier(n_estimators=50),
    "random_forest_100": ensemble.RandomForestClassifier(n_estimators=100),
    "random_forest_500": ensemble.RandomForestClassifier(n_estimators=500),
    "nearest_centroid": neighbors.NearestCentroid(),
    "naive_bayes": naive_bayes.GaussianNB(),
    "linear_svm": svm.LinearSVC(),
    "linear_svm_001": svm.LinearSVC(C=0.01, max_iter=10_000),
    "linear_svm_01": svm.LinearSVC(C=0.1, max_iter=10_000),
    "linear_svm_1": svm.LinearSVC(C=1.0, max_iter=10_000),
    "linear_svm_10": svm.LinearSVC(C=10.0, max_iter=10_000),
}
