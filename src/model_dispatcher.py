#!python3

from sklearn import tree, ensemble

models = {
    "decision_tree": tree.DecisionTreeClassifier(),
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    "random_forest": ensemble.RandomForestClassifier()
}
