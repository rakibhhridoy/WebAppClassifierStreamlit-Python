import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    st.title('Binary Classification Web App')
    st.sidebar.title('Binary Classifier App')
    st.markdown("Are Your Mashroom poisonous?")
    st.sidebar.markdown("Are Your Mashroom poisonous?")

    @st.cache(persist = True)
    def load_data():
        df = pd.read_csv('/home/rhyme/Desktop/Project/mushrooms.csv')
        le = LabelEncoder()

        for col in df.columns:
            df[col] = le.fit_transform(df[col])

        return df
    
    @st.cache(persist= True)
    def split(df):
        y = df.type
        x = df.drop(columns= ['type'])
        xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.3, random_state = 0)

        return xtrain, xtest, ytrain, ytest


    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            plot_confusion_matrix(model, xtest, ytest, display_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            plot_roc_curve(model, xtest, ytest)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve') 
            plot_precision_recall_curve(model, xtest, ytest)
            st.pyplot()





    df = load_data()
    xtrain, xtest, ytrain, ytest = split(df)
    class_names = ['edible','poisonous']
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine(SVM)", "Logistic Regression", "Random Classifier"))
    

    if classifier == "Support Vector Machine(SVM)":
        st.sidebar.subheader("Model HyperParameter")
        c = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0, step = 0.01, key= 'c')
        kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient", ('scale', 'auto'), key = 'gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader('Support Vector Machine (SVM) Results')
            model = SVC(C= c, kernel=kernel,gamma= gamma)
            model.fit(xtrain, ytrain)
            accuracy = model.score(xtest, ytest)
            ypred = model.predict(xtest)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision", precision_score(ytest, ypred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(ytest, ypred, labels = class_names).round(2))
            plot_metrics(metrics)



    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model HyperParameter")
        c_lr = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0, step = 0.01, key= 'c_lr')
        max_iter = st.sidebar.slider("maximum number of iteration", 100, 500, key = 'max_iter')


        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader('Logistic Regression Results')
            model = LogisticRegression(C= c_lr, max_iter= max_iter)
            model.fit(xtrain, ytrain)
            accuracy = model.score(xtest, ytest)
            ypred = model.predict(xtest)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision", precision_score(ytest, ypred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(ytest, ypred, labels = class_names).round(2))
            plot_metrics(metrics)


    if classifier == "Random Classifier":
        st.sidebar.subheader("Model HyperParameter")
        n_estimator = st.sidebar.number_input("Number of Trees in the forest", 100,5000, step =10, key= 'n_estimator')
        max_depth = st.sidebar.slider("Maximum Depth of the tree", 1, 20, step = 1,key = 'max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key ='bootstrap')


        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader('Random Forest Result')
            model = RandomForestClassifier(n_estimators=n_estimator, max_depth= max_depth, bootstrap=bootstrap)
            model.fit(xtrain, ytrain)
            accuracy = model.score(xtest, ytest)
            ypred = model.predict(xtest)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision", precision_score(ytest, ypred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(ytest, ypred, labels = class_names).round(2))
            plot_metrics(metrics)











    if st.sidebar.checkbox('Show raw data', False):
        st.subheader('Mushroom Data Set (Cassification')
        st.write(df)






if __name__ == '__main__':
    main()


