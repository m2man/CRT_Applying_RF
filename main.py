import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn import datasets
import pandas as pd

# Set the parameters by cross-validation for RF
tuned_parameters = {'n_estimators': [100, 200, 300, 400], 
                    'min_samples_split': [0.01, 0.05, 0.1],
                    'min_samples_leaf': [0.01, 0.005, 0.5]}

run_time = 100
    
def run_iris(run_time=100):
    print('----- IRIS DATASET -----')
    # Load dataset
    iris = datasets.load_iris()
    X = iris['data']
    Y = iris['target']
    
    rtt_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_acc_test = np.zeros(len(rtt_list))
    mean_var_test = np.zeros(len(rtt_list))
    mean_acc_train = np.zeros(len(rtt_list))
    mean_var_train = np.zeros(len(rtt_list))
    
    for idx_rtt, rtt in enumerate(rtt_list):
        # Define ratio between testing and training data
        # rtt = 0.1 # test-to-train ratio
        print(f"Processing rtt={rtt} ...")
        test_size = rtt * 1 / (1+rtt) 

        acc_test_list = np.zeros(run_time)
        acc_train_list = np.zeros(run_time)

        for idx_run in tqdm(range(run_time)):
            # Split data
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1509+idx_run)
            train_idx, test_idx = next(sss.split(X, Y))
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]

            # Cross validate to find best hyper parameters
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509), 
                               tuned_parameters, cv=5, verbose=0)
            clf.fit(X_train, Y_train)

            # Train again with best hyperparameter
            clf = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509, 
                                         min_samples_leaf = clf.best_params_['min_samples_leaf'],
                                         n_estimators = clf.best_params_['n_estimators'],
                                         min_samples_split = clf.best_params_['min_samples_split'])
            clf.fit(X_train, Y_train)

            # Prediction
            Y_pred_test = clf.predict(X_test)
            Y_pred_train = clf.predict(X_train)
            acc_test = accuracy_score(Y_test, Y_pred_test)
            acc_train = accuracy_score(Y_train, Y_pred_train)
            acc_test_list[idx_run] = acc_test
            acc_train_list[idx_run] = acc_train
        m_a_test = np.mean(acc_test_list)
        m_a_train = np.mean(acc_train_list)
        m_v_test = np.var(acc_test_list)
        m_v_train = np.var(acc_train_list)
        mean_acc_test[idx_rtt] = round(m_a_test,4)
        mean_acc_train[idx_rtt] = round(m_a_train,4)
        mean_var_test[idx_rtt] = round(m_v_test,4)
        mean_var_test[idx_rtt] = round(m_v_train,4)
    print("Mean Accuracy Test")
    print(mean_acc_test)
    print("Mean Accuracy Train")
    print(mean_acc_train)
    print("Var Accuracy Test")
    print(mean_var_test)
    print("Var Accuracy Train")
    print(mean_var_train)

def run_planning(run_time=100):
    # Load dataset
    print('----- PLANNING DATASET -----')
    data = pd.read_csv('Dataset/planning_dataset.txt', sep="\t", header=None)
    data.columns = [f"f{x}" for x in range(14)]
    data = data.drop(columns=['f13']) # last column is redundant
    data.f12 = data.f12.astype('int64')
    X = data.iloc[:,:-1].to_numpy()
    Y = data.iloc[:,-1].to_numpy()

    rtt_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_acc_test = np.zeros(len(rtt_list))
    mean_var_test = np.zeros(len(rtt_list))
    mean_acc_train = np.zeros(len(rtt_list))
    mean_var_train = np.zeros(len(rtt_list))
    
    for idx_rtt, rtt in enumerate(rtt_list):
        # Define ratio between testing and training data
        # rtt = 0.1 # test-to-train ratio
        print(f"Processing rtt={rtt} ...")
        test_size = rtt * 1 / (1+rtt) 

        acc_test_list = np.zeros(run_time)
        acc_train_list = np.zeros(run_time)

        for idx_run in tqdm(range(run_time)):
            # Split data
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1509+idx_run)
            train_idx, test_idx = next(sss.split(X, Y))
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]

            # Cross validate to find best hyper parameters
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509), 
                               tuned_parameters, cv=5, verbose=0)
            clf.fit(X_train, Y_train)

            # Train again with best hyperparameter
            clf = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509, 
                                         min_samples_leaf = clf.best_params_['min_samples_leaf'],
                                         n_estimators = clf.best_params_['n_estimators'],
                                         min_samples_split = clf.best_params_['min_samples_split'])
            clf.fit(X_train, Y_train)

            # Prediction
            Y_pred_test = clf.predict(X_test)
            Y_pred_train = clf.predict(X_train)
            acc_test = accuracy_score(Y_test, Y_pred_test)
            acc_train = accuracy_score(Y_train, Y_pred_train)
            acc_test_list[idx_run] = acc_test
            acc_train_list[idx_run] = acc_train
        m_a_test = np.mean(acc_test_list)
        m_a_train = np.mean(acc_train_list)
        m_v_test = np.var(acc_test_list)
        m_v_train = np.var(acc_train_list)
        mean_acc_test[idx_rtt] = round(m_a_test,4)
        mean_acc_train[idx_rtt] = round(m_a_train,4)
        mean_var_test[idx_rtt] = round(m_v_test,4)
        mean_var_test[idx_rtt] = round(m_v_train,4)
    print("Mean Accuracy Test")
    print(mean_acc_test)
    print("Mean Accuracy Train")
    print(mean_acc_train)
    print("Var Accuracy Test")
    print(mean_var_test)
    print("Var Accuracy Train")
    print(mean_var_train)

def run_seeds(run_time=100):
    # Load dataset
    print('----- SEEDS DATASET -----')
    data = pd.read_csv('Dataset/seeds_dataset.txt', sep="\t", header=None)
    data.columns = [f"f{x}" for x in range(8)]
    data.f7 = data.f7.astype('int64')
    X = data.iloc[:,:-1].to_numpy()
    Y = data.iloc[:,-1].to_numpy()

    rtt_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_acc_test = np.zeros(len(rtt_list))
    mean_var_test = np.zeros(len(rtt_list))
    mean_acc_train = np.zeros(len(rtt_list))
    mean_var_train = np.zeros(len(rtt_list))
    
    for idx_rtt, rtt in enumerate(rtt_list):
        # Define ratio between testing and training data
        # rtt = 0.1 # test-to-train ratio
        print(f"Processing rtt={rtt} ...")
        test_size = rtt * 1 / (1+rtt) 

        acc_test_list = np.zeros(run_time)
        acc_train_list = np.zeros(run_time)

        for idx_run in tqdm(range(run_time)):
            # Split data
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1509+idx_run)
            train_idx, test_idx = next(sss.split(X, Y))
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]

            # Cross validate to find best hyper parameters
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509), 
                               tuned_parameters, cv=5, verbose=0)
            clf.fit(X_train, Y_train)

            # Train again with best hyperparameter
            clf = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509, 
                                         min_samples_leaf = clf.best_params_['min_samples_leaf'],
                                         n_estimators = clf.best_params_['n_estimators'],
                                         min_samples_split = clf.best_params_['min_samples_split'])
            clf.fit(X_train, Y_train)

            # Prediction
            Y_pred_test = clf.predict(X_test)
            Y_pred_train = clf.predict(X_train)
            acc_test = accuracy_score(Y_test, Y_pred_test)
            acc_train = accuracy_score(Y_train, Y_pred_train)
            acc_test_list[idx_run] = acc_test
            acc_train_list[idx_run] = acc_train
        m_a_test = np.mean(acc_test_list)
        m_a_train = np.mean(acc_train_list)
        m_v_test = np.var(acc_test_list)
        m_v_train = np.var(acc_train_list)
        mean_acc_test[idx_rtt] = round(m_a_test,4)
        mean_acc_train[idx_rtt] = round(m_a_train,4)
        mean_var_test[idx_rtt] = round(m_v_test,4)
        mean_var_test[idx_rtt] = round(m_v_train,4)
    print("Mean Accuracy Test")
    print(mean_acc_test)
    print("Mean Accuracy Train")
    print(mean_acc_train)
    print("Var Accuracy Test")
    print(mean_var_test)
    print("Var Accuracy Train")
    print(mean_var_train)

def run_sonar(run_time=100):
    # Load dataset
    print('----- SONAR DATASET -----')
    data = pd.read_csv('Dataset/sonar_all.txt', sep=",", header=None)
    data.columns = [f"f{x}" for x in range(61)]
    data.f60 = data.f60.astype('category')
    data.f60 = data.f60.cat.codes
    X = data.iloc[:,:-1].to_numpy()
    Y = data.iloc[:,-1].to_numpy()

    rtt_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_acc_test = np.zeros(len(rtt_list))
    mean_var_test = np.zeros(len(rtt_list))
    mean_acc_train = np.zeros(len(rtt_list))
    mean_var_train = np.zeros(len(rtt_list))
    
    for idx_rtt, rtt in enumerate(rtt_list):
        # Define ratio between testing and training data
        # rtt = 0.1 # test-to-train ratio
        print(f"Processing rtt={rtt} ...")
        test_size = rtt * 1 / (1+rtt) 

        acc_test_list = np.zeros(run_time)
        acc_train_list = np.zeros(run_time)

        for idx_run in tqdm(range(run_time)):
            # Split data
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1509+idx_run)
            train_idx, test_idx = next(sss.split(X, Y))
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]

            # Cross validate to find best hyper parameters
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509), 
                               tuned_parameters, cv=5, verbose=0)
            clf.fit(X_train, Y_train)

            # Train again with best hyperparameter
            clf = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509, 
                                         min_samples_leaf = clf.best_params_['min_samples_leaf'],
                                         n_estimators = clf.best_params_['n_estimators'],
                                         min_samples_split = clf.best_params_['min_samples_split'])
            clf.fit(X_train, Y_train)

            # Prediction
            Y_pred_test = clf.predict(X_test)
            Y_pred_train = clf.predict(X_train)
            acc_test = accuracy_score(Y_test, Y_pred_test)
            acc_train = accuracy_score(Y_train, Y_pred_train)
            acc_test_list[idx_run] = acc_test
            acc_train_list[idx_run] = acc_train
        m_a_test = np.mean(acc_test_list)
        m_a_train = np.mean(acc_train_list)
        m_v_test = np.var(acc_test_list)
        m_v_train = np.var(acc_train_list)
        mean_acc_test[idx_rtt] = round(m_a_test,4)
        mean_acc_train[idx_rtt] = round(m_a_train,4)
        mean_var_test[idx_rtt] = round(m_v_test,4)
        mean_var_test[idx_rtt] = round(m_v_train,4)
    print("Mean Accuracy Test")
    print(mean_acc_test)
    print("Mean Accuracy Train")
    print(mean_acc_train)
    print("Var Accuracy Test")
    print(mean_var_test)
    print("Var Accuracy Train")
    print(mean_var_train)
    
def run_wine(run_time=100):
    # Load dataset
    print('----- WINE DATASET -----')
    data = pd.read_csv('Dataset/wine_data.txt', sep=",", header=None)
    data.columns = [f"f{x}" for x in range(14)]
    data.f0 = data.f0.astype('int64')
    X = data.iloc[:,1:].to_numpy()
    Y = data.iloc[:,0].to_numpy()

    rtt_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean_acc_test = np.zeros(len(rtt_list))
    mean_var_test = np.zeros(len(rtt_list))
    mean_acc_train = np.zeros(len(rtt_list))
    mean_var_train = np.zeros(len(rtt_list))
    
    for idx_rtt, rtt in enumerate(rtt_list):
        # Define ratio between testing and training data
        # rtt = 0.1 # test-to-train ratio
        print(f"Processing rtt={rtt} ...")
        test_size = rtt * 1 / (1+rtt) 

        acc_test_list = np.zeros(run_time)
        acc_train_list = np.zeros(run_time)

        for idx_run in tqdm(range(run_time)):
            # Split data
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1509+idx_run)
            train_idx, test_idx = next(sss.split(X, Y))
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]

            # Cross validate to find best hyper parameters
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509), 
                               tuned_parameters, cv=5, verbose=0)
            clf.fit(X_train, Y_train)

            # Train again with best hyperparameter
            clf = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=1509, 
                                         min_samples_leaf = clf.best_params_['min_samples_leaf'],
                                         n_estimators = clf.best_params_['n_estimators'],
                                         min_samples_split = clf.best_params_['min_samples_split'])
            clf.fit(X_train, Y_train)

            # Prediction
            Y_pred_test = clf.predict(X_test)
            Y_pred_train = clf.predict(X_train)
            acc_test = accuracy_score(Y_test, Y_pred_test)
            acc_train = accuracy_score(Y_train, Y_pred_train)
            acc_test_list[idx_run] = acc_test
            acc_train_list[idx_run] = acc_train
        m_a_test = np.mean(acc_test_list)
        m_a_train = np.mean(acc_train_list)
        m_v_test = np.var(acc_test_list)
        m_v_train = np.var(acc_train_list)
        mean_acc_test[idx_rtt] = round(m_a_test,4)
        mean_acc_train[idx_rtt] = round(m_a_train,4)
        mean_var_test[idx_rtt] = round(m_v_test,4)
        mean_var_test[idx_rtt] = round(m_v_train,4)
    print("Mean Accuracy Test")
    print(mean_acc_test)
    print("Mean Accuracy Train")
    print(mean_acc_train)
    print("Var Accuracy Test")
    print(mean_var_test)
    print("Var Accuracy Train")
    print(mean_var_train)
    
# run_iris(run_time=100)
# run_planning(run_time=100)
# run_seeds(run_time=100)
# run_sonar(run_time=100)
run_wine(run_time=100)
