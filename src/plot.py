import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_toml_paths

# Settings
sns.set(style="darkgrid")
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

data = load_toml_paths()


def plot_heatmap(df):
    """Plots a heatmap using seaborn

    Keyword arguments:
    df - dataframe

    Return: dataframe
    """
    df = df.dropna('columns')  # drop columns with NaN
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]]
    corr = df.corr()

    plt.figure(figsize=(15, 12))
    sns.heatmap(corr)  # Refer figs/plot1.png
    plt.savefig(data['heatmap1'])

    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))

    # plot heat map
    # Refer figs/plot2.png
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.savefig(data['heatmap2'])

    return df


def plot_training_accuracies(classifiers, training_accuracies_models):
    """Plots model training accuracies

    Keyword arguments:
    classifiers -- List of classifiers names
    training_accuracies_models -- Accuracy with training data

    Return: None
    """
    fig = plt.figure()
    fig.suptitle("Training Accuracies")
    ax = fig.add_subplot(111)
    ax.set_xticklabels(classifiers)

    plt.bar(classifiers, training_accuracies_models)
    plt.ylim(0.8, 1)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')

    fig.savefig(data['training_accuracy_models'], bbox_inches='tight')


def plot_testing_accuracies(classifiers, testing_accuracies_models):
    """Plots model testing accuracies

    Keyword arguments:
    classifiers -- List of classifiers names
    testing_accuracies_models -- Accuracy with testing data

    Return: None
    """
    fig = plt.figure()
    fig.suptitle("Testing Accuracies")
    ax = fig.add_subplot(111)
    ax.set_xticklabels(classifiers)

    plt.bar(classifiers, testing_accuracies_models)
    plt.ylim(0.8, 1)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')

    fig.savefig(data['testing_accuracy_models'], bbox_inches='tight')


def plot_training_ensembles_accuracies(ensembles,
                                       training_accuracies_ensembles):
    """Plots ensembles training accuracies

    Keyword arguments:
    classifiers -- List of classifiers names
    training_accuracies_ensembles -- Accuracy with training data

    Return: None
    """
    fig = plt.figure()
    fig.suptitle("Training Accuracies")
    ax = fig.add_subplot(111)
    ax.set_xticklabels(ensembles)

    plt.bar(ensembles, training_accuracies_ensembles)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')

    fig.savefig(data['training_ensembles_accuracies'], bbox_inches='tight')


def plot_testing_ensembles_accuracies(ensembles, testing_accuracies_ensembles):
    """Plots ensembles testing accuracies

    Keyword arguments:
    classifiers -- List of classifiers names
    testing_accuracies_ensembles -- Accuracy with testing data

    Return: None
    """
    fig = plt.figure()
    fig.suptitle("Testing Accuracies")
    ax = fig.add_subplot(111)
    ax.set_xticklabels(ensembles)

    plt.bar(ensembles, testing_accuracies_ensembles)
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')

    fig.savefig(data['testing_ensembles_accuracies'], bbox_inches='tight')
