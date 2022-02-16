from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
import streamlit as st
import numpy as np


st.title("Streamlit Testing App")

# st.write(""" # Type 1 Heading
# """)

dataset_name = st.sidebar.selectbox("Select Dataset", ('Iris','Wine','Breast Cancer'))

st.write(f'## {dataset_name} Dataset')

classifier_name = st.sidebar.selectbox("Select Classifier", ('Logistic Regression', 'SVM', 'Random Forest'))

st.write(f'## {classifier_name} Algorithm')

def get_dataset(name):
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    
    X = data.data
    y = data.target

    return X,y

X,y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_param_model(clf_name):
    params = dict()
    if clf_name == 'Logistic Regression':

        penalty = st.sidebar.selectbox('Penalty Parameter', ('none', 'l1', 'l2', 'elasticnet'))
        params['penalty'] = penalty

        solver = st.sidebar.selectbox('Solver Parameter', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
        params['solver'] = solver

    elif clf_name == 'SVM':

        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C

    else:

        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth

        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators

    return params

params = add_param_model(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(penalty=params['penalty'], solver=params['solver'], random_state=42)

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42)

    else:
        clf =  SVC(C=params['C'])

    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_pred, y_test)

# report = classification_report(y_pred, y_test)

matrix = confusion_matrix(y_pred, y_test)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
# st.write(f'Report =', report)
st.write(f'Matrix =', matrix)

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)