from src.plot import plot_heatmap
from src.plot import plot_testing_accuracies
from src.plot import plot_training_accuracies
from src.plot import plot_testing_ensembles_accuracies
from src.plot import plot_training_ensembles_accuracies
from src.test import test_pipeline
from src.utils import data_cleaning, question2
from src.utils import data_preparation
from src.utils import ensembles_training
from src.utils import feature_selection
from src.utils import hyperparameter_tuning
from src.utils import model_selection
from src.utils import model_training
from src.utils import model_validation_evaluation
from src.utils import read_data
from src.utils import save_correlation_results

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

PATH = 'data/data.csv'


kdd_data = read_data(PATH)

# Data Cleaning
df = data_cleaning(kdd_data)


# Exploratory data analysis
df = plot_heatmap(df)
save_correlation_results(df)


# feature selection
X, y, df, kbest = feature_selection(df)


# Data Preparation for Modelling
X_train, X_test, Y_train, Y_test = data_preparation(X, y, df, kbest)

# Model Building
classifiers, training_accuracies_models, testing_accuracies_models =\
                 model_training(X_train, X_test, Y_train, Y_test)

# Training Accuracies
plot_training_accuracies(classifiers, training_accuracies_models)

# Testing Accuracies
plot_testing_accuracies(classifiers, testing_accuracies_models)

# Ensembles
classifiers, training_accuracies_ensembles, testing_accuracies_ensembles = \
                  ensembles_training(X_train, X_test, Y_train, Y_test)

# Ensembles Training Accuracies
plot_training_ensembles_accuracies(classifiers, training_accuracies_ensembles)

# Ensembles Testing Accuracies
plot_testing_ensembles_accuracies(classifiers, testing_accuracies_ensembles)

# Model selection
model = model_selection(X_train, Y_train)

# Hyperparameter tuning
pipeline, x_train, x_test, y_train, y_test = hyperparameter_tuning(X_train,
                                                                   Y_train,
                                                                   df, kbest)

# Model Validation & Evaluation
model_validation_evaluation(pipeline, x_test, y_test)

# Tests pipeline
test_pipeline(pipeline)

# Question2
# AUC values
question2(kdd_data)
