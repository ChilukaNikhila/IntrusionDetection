import dataframe_image as dfi
import numpy as np
import pandas as pd
import pickle
import toml
# Libraries for Data Pre Processing
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Libraries for Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Libraries for Model Selection
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
# Deployment
from sklearn.pipeline import Pipeline
# Settings
pd.set_option('display.max_columns', None)
np.set_printoptions(precision=3)

# Mapping attack field under attck class
MAPPING = {'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe',
           'portsweep': 'Probe', 'saint': 'Probe', 'mscan': 'Probe',
           'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS',
           'neptune': 'DoS', 'smurf': 'DoS', 'mailbomb': 'DoS',
           'udpstorm': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
           'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
           'buffer_overflow': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
           'sqlattack': 'U2R', 'httptunnel': 'U2R',
           'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L',
           'warezmaster': 'R2L', 'warezclient': 'R2L', 'imap': 'R2L',
           'spy': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'snmpguess': 'R2L',
           'worm': 'R2L', 'snmpgetattack': 'R2L',
           'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
           'normal': 'Normal'}

# To load toml paths


def load_toml_paths():
    data = toml.load("pyproject.toml")
    return data['data']['paths']


data = load_toml_paths()

# To save numpy files


def save_numpy_data(filename, variable):
    np.save(filename, variable)

# To save Dataframes


def save_dataframe(filename, df):
    # adding a gradient based on values in cell
    df_styled = df.style.background_gradient()
    dfi.export(df_styled, filename)

# To save txt files


def save_text(filename, text):
    with open(filename, "w") as text_file:
        text_file.write(text)

# To save models


def save_model(pipeline):
    pickle.dump(pipeline, open(data['pipeline'], 'wb'))


def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model

# To read data


def read_data(path):
    """Read the data

    Keyword arguments:
    Path -- path to the data.csv

    Return: None
    """
    kdd_data = pd.read_csv(path)
    return kdd_data

# To clean data


def data_cleaning(kdd_data):
    """Cleans the data

    Keyword arguments:
    kdd_data -- dataframe

    Return: dataframe
    """

    # Apply attack class mappings to the dataset
    kdd_data['attack_class'] = kdd_data['label'].apply(lambda v: MAPPING[v])

    # Drop label field from both train and test data
    kdd_data.drop(['label'], axis=1, inplace=True)

    # Making a duplicate in df
    df = kdd_data

    kdd_data['attack_class'].count()  # value: 494020

    kdd_data['attack_class'].value_counts()
    """
    Output:

    DoS       391458
    Normal     97277
    Probe       4107
    R2L         1126
    U2R           52
    Name: attack_class, dtype: int64
    """

    df.isnull().sum()

    df['num_outbound_cmds'].value_counts()
    """
    Output:

    0    494020
    Name: num_outbound_cmds, dtype: int64
    """

    # 'num_outbound_cmds' field has all 0 values
    #  Hence, it will be removed from both train and test dataset since it is a redundant field.
    df.drop(['num_outbound_cmds'], axis=1, inplace=True)

    return df


def save_correlation_results(df):
    """Saves correlation results

    Keyword arguments:
    argument -- description

    Return: return_description
    """
    results = {
        'num_root_X_num_compromised': df['num_root'].corr(df['num_compromised']),
        'srv_serror_rate_X_serror_rate': df['srv_serror_rate'].corr(df['serror_rate']),
        'srv_count_X_count': df['srv_count'].corr(df['count']),
        'srv_rerror_rate_X_rerror_rate': df['srv_rerror_rate'].corr(df['rerror_rate']),
        'dst_host_same_srv_rate_X_dst_host_srv_count': df['dst_host_same_srv_rate'].corr(df['dst_host_srv_count']),
        'dst_host_srv_serror_rate_X_dst_host_serror_rate': df['dst_host_srv_serror_rate'].corr(df['dst_host_serror_rate']),
        'dst_host_srv_rerror_rate_X_dst_host_rerror_rate': df['dst_host_srv_rerror_rate'].corr(df['dst_host_rerror_rate']),
        'dst_host_same_srv_rate_X_same_srv_rate': df['dst_host_same_srv_rate'].corr(df['same_srv_rate']),
        'dst_host_srv_count_X_same_srv_rate': df['dst_host_srv_count'].corr(df['same_srv_rate']),
        'dst_host_same_src_port_rate_X_srv_count': df['dst_host_same_src_port_rate'].corr(df['srv_count']),
        'dst_host_serror_rate_X_serror_rate': df['dst_host_serror_rate'].corr(df['serror_rate']),
        'dst_host_serror_rate_X_srv_serror_rate': df['dst_host_serror_rate'].corr(df['srv_serror_rate']),
        'dst_host_srv_serror_rate_X_serror_rate': df['dst_host_srv_serror_rate'].corr(df['serror_rate']),
        'dst_host_srv_serror_rate_X_srv_serror_rate': df['dst_host_srv_serror_rate'].corr(df['srv_serror_rate']),
        'dst_host_rerror_rate_X_rerror_rate': df['dst_host_rerror_rate'].corr(df['rerror_rate']),
        'dst_host_rerror_rate_X_srv_rerror_rate': df['dst_host_rerror_rate'].corr(df['srv_rerror_rate']),
        'dst_host_srv_rerror_rate_X_rerror_rate': df['dst_host_srv_rerror_rate'].corr(df['rerror_rate']),
        'dst_host_srv_rerror_rate_X_srv_rerror_rate': df['dst_host_srv_rerror_rate'].corr(df['srv_rerror_rate'])
    }

    save_numpy_data(data['correlation_results'], results)


def calculate_standard_deviation(df):
    df_std = df.std()
    df_std = df_std.sort_values(ascending=True)
    save_numpy_data(data['features_std'], df_std)


def remove_correlated_attributes(df):
    """Removes highly correlated values

    Keyword arguments:
    df -- dataframe

    Return: dataframe and Input, Output attributes
    """
    # This variable is highly correlated with num_compromised
    # And should be ignored for analysis.
    # (Correlation = 0.9938277978738366)
    df.drop('num_root', axis=1, inplace=True)

    # This variable is highly correlated with serror_rate
    # And should be ignored for analysis.
    # (Correlation = 0.9983615072725952)
    df.drop('srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate
    # should be ignored for analysis.
    # (Correlation = 0.9947309539817937)
    df.drop('srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_serror_rate
    # (Correlation = 0.9993041091850098)
    df.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate
    # (Correlation = 0.9869947924956001)
    df.drop('dst_host_serror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate
    # (Correlation = 0.9821663427308375)
    df.drop('dst_host_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with rerror_rate
    # (Correlation = 0.9851995540751249)
    df.drop('dst_host_srv_rerror_rate', axis=1, inplace=True)

    # This variable is highly correlated with srv_rerror_rate
    # (Correlation = 0.9865705438845669)
    df.drop('dst_host_same_srv_rate', axis=1, inplace=True)

    calculate_standard_deviation(df)

    df.drop('service', axis=1, inplace=True)

    # protocol_type feature mapping
    pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
    df['protocol_type'] = df['protocol_type'].map(pmap)

    # flag feature mapping
    fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4,
            'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}
    df['flag'] = df['flag'].map(fmap)

    X = df.drop(['attack_class'],axis=1)
    y = df['attack_class']

    return X, y, df


def feature_selection(df):
    """Features are selected

    Keyword arguments:
    df -- dataframe

    Return: Input and Ouput attributes, selected features with dataframe
    """
    X, y, df = remove_correlated_attributes(df)  # Removes highly correlated values

    # Selecting Features using chi square as Output is categorical varible
    bestfeatures = SelectKBest(score_func=chi2, k=8)
    fit = bestfeatures.fit(X, y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']   # naming the dataframe columns
    print(featureScores.nlargest(8, 'Score'))
    save_numpy_data(data['features_scores'], featureScores)

    kbest = ['src_bytes', 'dst_bytes', 'duration', 'count',
             'srv_count', 'dst_host_count', 'hot', 'dst_host_srv_count']

    save_numpy_data(data['selected_columns'], kbest)

    return X, y, df, kbest


def data_preparation(X, y, df, kbest):
    """Making Data Ready for Model Training & Testingary_line

    Keyword arguments:
    X -- Input attributes
    y -- Output attributes
    df -- dataframe
    kbest -- selected features

    Return: return_description
    """
    X = df[kbest]
    y = df[['attack_class']]
    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    # Split test and train data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train.shape, X_test.shape)  # (330993, 8) (163027, 8)
    print(Y_train.shape, Y_test.shape)  # (330993, 1) (163027, 1)

    save_numpy_data(data['selected_features'], X)

    return X_train, X_test, Y_train, Y_test


def define_models():
    models = []
    models.append(('Logitsic', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('Decision Tree', DecisionTreeClassifier()))
    models.append(('SVM', SVC()))
    models.append(('Naive Bayes', GaussianNB()))

    return models


def model_training(X_train, X_test, Y_train, Y_test):
    """Model Building, Validation & Evaluation

    Keyword arguments:
    X_train -- Input training data
    X_test -- Input testing data
    Y_train -- Output training data
    Y_test -- Output testing data

    Return: classifiers, training_accuracies_models, testing_accuracies_models
    """
    # testing on multiple models
    models = define_models()

    # results
    training_accuracies_models = []
    testing_accuracies_models = []
    classifiers = []
    result = ""
    for name, model in models:
        classifiers.append(name)
        model.fit(X_train, Y_train)
        training_accuracies_models.append(model.score(X_train, Y_train))
        testing_accuracies_models.append(model.score(X_test, Y_test))
        confusion_matrix = metrics.confusion_matrix(Y_train,
                                                    model.predict(X_train))
        classification = metrics.classification_report(Y_train,
                                                       model.predict(X_train))
        print()
        print('============================== {} Model Test Results ==========\
        ===================='.format(name))
        print()
        print("Confusion matrix:" "\n", confusion_matrix)
        print()
        print("Classification report:" "\n", classification)
        print()

        result += "\n" + '============================== {} Model Test Results \
            =============================='.format(name) + "\n" + \
            "Confusion matrix:" + "\n" + str(confusion_matrix) + "\n" +\
            "Classification report:" + "\n" + str(classification) + "\n"

    save_text(data['model_test_results'], result)

    Compare_Models = pd.DataFrame(list(zip(classifiers, training_accuracies_models, testing_accuracies_models)), columns=['Classifer', 'Training Accuracy', 'Testing Accruacy'])
    save_dataframe(data['models_comparision'], Compare_Models)

    return classifiers, training_accuracies_models, testing_accuracies_models


def define_ensembles():
    ea = []
    ea.append(('AdaBoost', AdaBoostClassifier()))
    ea.append(('RandF', RandomForestClassifier()))
    ea.append(('XGB', XGBClassifier()))

    return ea


def ensembles_training(X_train, X_test, Y_train, Y_test):
    """Ensembles

    Keyword arguments:
    X_train -- Input training data
    X_test -- Input testing data
    Y_train -- Output training data
    Y_test -- Output testing data

    Return: classifiers, training_accuracies_models, testing_accuracies_models
    """
    ea = define_ensembles()
    # results
    training_accuracies_ensembles = []
    testing_accuracies_ensembles = []
    ensembles = []
    result = ""
    for name, model in ea:
        ensembles.append(name)
        model.fit(X_train, Y_train)
        training_accuracies_ensembles.append(model.score(X_train, Y_train))
        testing_accuracies_ensembles.append(model.score(X_test, Y_test))
        confusion_matrix = metrics.confusion_matrix(Y_test, model.predict(X_test))
        classification = metrics.classification_report(Y_test, model.predict(X_test))
        print()
        print('============================== {} Model Test Results ===\
        ==========================='.format(name))
        print()
        print("Confusion matrix:" "\n", confusion_matrix)
        print()
        print("Classification report:" "\n", classification)
        print()

        result += "\n" + '============================== {} Model Test Results\
        =============================='.format(name) + "\n" + \
            "Confusion matrix:" + "\n" + str(confusion_matrix) + "\n" +\
            "Classification report:" + "\n" + str(classification) + "\n"

    save_text(data['ensemble_test_results'], result)

    Compare_Ensembles = pd.DataFrame(list(zip(ensembles, training_accuracies_ensembles, testing_accuracies_ensembles)),columns =['Classifer', 'Training Accuracy','Testing Accruacy'])
    save_dataframe(data['ensembles_comparision'], Compare_Ensembles)

    return ensembles, training_accuracies_ensembles, testing_accuracies_ensembles


def model_selection(X_train, Y_train):
    """Selects model

    Keyword arguments:
    X_train -- Input training data
    Y_train -- Output training data

    Return: a classification model
    """

    decisiontree = DecisionTreeClassifier()
    decisiontree.fit(X_train, Y_train)
    decisiontree.get_params()
    """
    Output:

    {'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': None,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_impurity_split': None,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'random_state': None,
    'splitter': 'best'}

    Max_Depth set to Nonewhich may lead to Overfitting.
    Gini is intended for continuous attributes
    Entropy is for attributes that occur in classes.
    So Gini is the Best Choice.
    """

    return decisiontree


def hyperparameter_tuning(X_train, Y_train, df, kbest):
    """Hyperparameter tuning

    Keyword arguments:
    X_train -- Input training data
    Y_train -- Output training data
    df -- dataframe
    kbest -- selected features

    Return: pipeline, x_train, x_test, y_train, y_test
    """
    param_dict = {
        "criterion": ['gini'],
        "max_depth": range(1, 10),
        "min_samples_split":  range(1, 10),
        "min_samples_leaf": range(1, 5)
    }
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_dict,
                        cv=10, verbose=1, n_jobs=-1)
    grid.fit(X_train, Y_train)

    x = df[kbest]
    y = df[['attack_class']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,
                                                        random_state=42)
    dec_tree = DecisionTreeClassifier()
    scaler = MinMaxScaler()

    # Setting Best Grid Parameters
    dec_tree.set_params(**grid.best_params_)

    pipeline = Pipeline(steps=[('scaler', scaler), ('dec_tree', dec_tree)])
    pipeline.fit(x_train, y_train)
    save_model(pipeline)

    return pipeline, x_train, x_test, y_train, y_test


def model_validation_evaluation(pipeline, x_test, y_test):
    """Validates and Evaluates Model

    Keyword arguments:
    pipeline -- pipeline
    x_test -- Input testing data
    y_test -- Output testing data

    Return: None
    """

    accuracy = pipeline.score(x_test, y_test)
    confusion_matrix = metrics.confusion_matrix(y_test,
                                                pipeline.predict(x_test))
    classification = metrics.classification_report(y_test,
                                                   pipeline.predict(x_test))

    print("Accuracy : ", accuracy)
    print("------------------------------")
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()

    result = "Accuracy: " + str(accuracy) + "\n" + "----------------------" +\
             "Confusion matrix:" + "\n" + str(confusion_matrix) + \
             "Classification report:" + "\n" + str(classification)

    save_text(data['model_validation'], result)

# Question2


def question2(kdd_data):
    # checking for the null values
    kdd_data.isnull().sum()
    # Mapping attack field to attack class
    # Mapping attack field under attck class
    MAPPING = {'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe',
               'portsweep': 'Probe', 'saint': 'Probe', 'mscan': 'Probe',
               'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS',
               'neptune': 'DoS', 'smurf': 'DoS', 'mailbomb': 'DoS',
               'udpstorm': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
               'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
               'buffer_overflow': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
               'sqlattack': 'U2R', 'httptunnel': 'U2R',
               'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L',
               'warezmaster': 'R2L', 'warezclient': 'R2L', 'imap': 'R2L',
               'spy': 'R2L', 'multihop': 'R2L', 'named': 'R2L',
               'snmpguess': 'R2L',
               'worm': 'R2L', 'snmpgetattack': 'R2L',
               'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
               'normal': 'Normal'}
    # Apply attack class mappings to the dataset
    kdd_data['attack_class'] = kdd_data['label'].apply(lambda v: MAPPING[v])
    # Mapping Four types of attacks under main attack class
    mapping = {'DoS': 'attack', 'U2R': 'attack', 'Probe': 'attack',
               'R2L': 'attack', 'Normal': 'Normal'}
    kdd_data['attack_class'] = kdd_data['attack_class'].apply(lambda v: mapping[v])
    # Drop label field from both train and test data
    kdd_data.drop(['label'], axis=1, inplace=True)
    # Making a duplicate in df
    df = kdd_data
    df = pd.DataFrame(kdd_data)
    df.attack_class.unique()
    # Measurements with categorical values
    categorical = ['protocol_type', 'service', 'flag', 'attack_class']
    le = LabelEncoder()
    df[categorical] = df[categorical].apply(le.fit_transform)
    # Storing label values in Y
    Y = df.attack_class
    X = df.drop(['attack_class'], axis=1)
    # Finding AUC values for every measurement
    auc = []
    col = []
    for i in X.columns:
        features = np.array(X[i])
        col.append(i)
        roc_values = roc_auc_score(Y, features, multi_class='ovo',
                                   average='weighted')
        auc.append(roc_values)
        # Creating data frame for storing values
    a = pd.DataFrame(auc)
    b = pd.DataFrame(col)
    final_values = pd.concat([a, b], axis=1)
    final_values.columns = ["AUC_Values", "Measurments"]
    d = abs(0.5 - final_values['AUC_Values'])
    final_values['difference'] = d
    final_values = final_values.sort_values(by="difference", ascending=False)
    sorted_auc = final_values[:10]
    print("The 10 leading Auc values after sorting\n", sorted_auc)
