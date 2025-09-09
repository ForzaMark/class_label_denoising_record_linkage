from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

TEACHER_MODELS = [
     ('rf', RandomForestClassifier()),
     ('svm', LinearSVC()),
     ('tree', DecisionTreeClassifier())
]