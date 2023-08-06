# Overview of the project 
The dataset used in this project is the NSL-KDD dataset. It is a standard dataset to train intrusion detection systems and make them ready to serve their purpose in network security. The main task is to select the best model for determining whether the network is under attack or not.

The following tasks are done on the data set:
Data Cleaning: Data is cleaned by mapping all the attacks under four attacks which are DoS, Probe, R2L, U2R  
Exploratory data analysis
feature selection:Selecting the Best and minimal number of features from the data set.
Data Preparation for Modelling :Making Data Ready for Model Training & Testing
Model Building for LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier,SVM, Naive Bayes
After the models are trained the test results are compared among the different models
Different ensembles are trained and the results are compared.The esembles I used are XGB,Randomforest, Adaboost.
After the models are trained, and performances are compared, best model is choosed.
After the model is selecyed ,parameters tuning is done for the selected model
Model Validation & Evaluation is done after the best parameters are selected.
Then testing is done on the pipeline to check how well the predictions are done.

## Dataset Description
The dataset used in this project is NSL-KDD dataset. It is a standard dataset to train intrusion detection system and make it ready for serving its purpose in network security.

## Total number ofsamples
The data set consists 494020 records

## Total number of measurements
It has 42 features out of which 41 features describe different parameters of the data traffic and the last feature is the label which specifies whether it is normal data packet or malicious data packet 

## Description of the measurements

Following are the different columns in the data set
Duration: It is the description of the amount of time spent for the connection. This field takes integers.
Service: This feature tells the type of network service used. It takes string value.
Flag: This feature tells the status of the connection whether it is normal, or any error has occurred or not. It takes string value.
Protocol: It describes the type of protocol used by the network for data transfer. It takes string values.
Dst bytes: This feature gives information about the number of bytes transferred from destination device to source device in the network. It has integer value.
Land: This field takes value 1 if the port numbers and IP addresses of source and destination are same else it takes the value 0. It takes two values either 0 or 1.
Urgent: If a data packet which is very important and needs to be transmitted before other data packets then the urgent bit in the required data packet has to be activated. Urgent feature describes the number of urgent data packets in the connection made between source and destination. It takes integer values.
Wrong fragment: It describes the number of wrong fragments in the data packets. It takes integer values.
Hot: This feature describes the number of hot indicators in the data like entering a system’s directory, entering a folder and making changes in that folder, creating and executing programs etc. It takes integer values.
Num failed logins: It gives the number of attempts failed to login to the connection. It 
takes integer values.
Logged in: It has the value 1 if the login status is successful else 0 if the login status is unsuccessful. It takes only two values either 0 or 1.
Num compromised: It gives the number of conditions compromised. It has integer values.
Root shell: It takes the value 1 if root shell is obtained else 0. It takes only two values either 0 or 1.
Num root: This feature gives the number of operations performed under root access. It takes integer values.
Num shells: It gives the number of shell prompts. It takes integer value.
Super user attempted: The root privileges or administrator privileges can be gained by using super user command i.e., “su root”. This feature gives the number of attempts made to gain root access. It takes integer value.
Num file creation: It gives the number of times a file is created. It takes integer value.
Num access files: It gives the number of operations performed on access control files. It takes integer value.
Num outbound cmds: It gives the number of outbound commands in the file transfer protocol session. It takes integer value.
Is guest login: It takes value 1 if the login is a guest else 0. It takes only two values either 0 or 1.
Is hot login: It takes value 1 if the login is root or administrator else 0. It takes only two values either 0 or 1.
Count: It gives information about the number of connections made to the same host device as the present connection in the past 2 seconds. It takes integer value.
Srv_count: It gives information about the number of connections to the same port number (service) as the present connection in the past 2 seconds. It takes integer value.
Serror_rate: It gives the percentage of connections that activated flags s0, s1, s2, s3 from the connections in count. It takes float value.
Srv_serror_rate: It gives the percentage of connections that activated flags s0, s1, s2, s3 from the connections in srv_count. It takes float value.
Srv_diff_host_rate: It gives the percentage of connections that are made to different destination devices from the connections in srv_count. It takes float value.
Dst_host_count: It gives the number of connections that have the same destination host internet protocol (IP) address. It takes integer value.
Dst_host_srv_count: It gives the number of connections that have the same port number (service). It takes integer value.
Dst_host_same_srv_rate: It gives the percentage of connections that are to same services from the connections in dst_host_count. It takes float value.
Dst_host_diff_srv_rate: It gives the percentage of connections that are to different services from the connections in dst_host_count. It takes float value.
Dst_host_same_src_port_rate: It gives the percentage of connections that are to same port of the source from the connections in dst_host_srv_count. It takes float value.
Dst_host_srv_diff_host_rate: It gives the percentage of connections that are made to different destination devices from the connections in dst_host_srv_count. It takes float value.
Dst_host_serror_rate: It gives the percentage of connections that activated the flags s0, s1, s2, s3 from the connections in dst_host_count. It takes float value.
Dst_host_srv_serror_rate: It gives the percentage of connections that activated the flags s0, s1, s2, s3 from the connections in dst_host_srv_count. It takes float value.
Dst_host_serror_rate: It gives the percentage of connections that activated the flag REJ from the connections in dst_host_count. It takes float value.
Dst_host_srv_serror_rate: It gives the percentage of connections that activated the flag REJ from the connections in dst_host_srv_count. It takes float value.
Label: It gives the information about the classification of the input network traffic. It takes string value, 


## Nature of the group of interest 
My group of interest is normal network from label column which are denoted by 1.
This set of group gives the status of network attack

## Nature of the group of not interest
My group of not interest is network under attack which is denoted by 0. 

As salary has null values, the coulmn is dropped.
so the dataset shape is (494020, 42)

The 10 leading Auc values after sorting
     AUC_Values                  Measurments  difference
      0.987981                        count    0.487981
      0.078385                    dst_bytes    0.421615
      0.893897                    srv_count    0.393897
      0.123865                protocol_type    0.376135
      0.144678                    logged_in    0.355322
      0.789080               dst_host_count    0.289080
      0.247162  dst_host_srv_diff_host_rate    0.252838
      0.748200  dst_host_same_src_port_rate    0.248200
      0.286744                      service    0.213256
      0.332776           srv_diff_host_rate    0.167224

Here, difference means the absolute difference between AUC_SCORES and 0.5 
By using difference the auc values are sorted farthest from (0.5).




##Dataset
The dataset used in this project is NSL-KDD dataset. It is a standard dataset to train intrusion detection system and make it ready for serving its purpose in network security. The dataset consists of most of the attacks over the internet and has a detailed information about each type of attack.
The dataset consists of a total of 125974 records of information related to data packets in the network traffic. It has 42 features out of which 41 features describe different parameters of the data traffic and the last feature is the label which specifies whether it is normal data packet or malicious data packet

The attacks specified in the dataset fall under four categories of attack types:
-Denial of service(DoS)
-User to root(U2R)
-Remote tolocal(R2L)
-Probe
DoS:Denial of service is not able to provide response to the requests made by the clients.Sub-categories of DoS in the dataset
neptune
pod
smurf
apache2
udpstorm
back
teardrop

U2R:When the attacker gains access to root permissions he/she can modify the system software configurations, steal confidential information, make the system look malicious in the network in which it resides, alter the encryption algorithms in the system root level etc
Sub-categories of U2R in the dataset
  load module
  perl
  buffer overflow
  SQL attack
  rootkit
  xterm
   ps

R2L:In this type of attack, access to host computer is gained by the attacker
Sub-categories of R2L in the dataset
 ftp_write
 guess_passwd
 imap
 multihop
 sendmail
 spy
 snmpguess
 warezclient
 warezmaster

 Probe:Probe attack is also known as site scanning attack
 Sub-categories of Probe in the dataset
  ipsweep
  mscan
  nmap
  saint
  satan
  portsweep

##Features in dataset:A total of 42 features are present in the dataset, out of which 41 features describe the details of the network traffic and the type of data. The last feature is a label that tells whether the network traffic is an attack or normal traffic.

## Data Cleaning

Data must be cleaned to improve the quality of Data. The Data Cleaning steps that are considered for the data is. 

1.	Mapping the attack classes. As there are subcategories of attack classes, they are mapped to the main category (Main classes are DoS, U2R, Probe, R2L)
2.	Dropping irrelevant columns: num_outbound_cmds' field has all 0 values. Hence, it will be removed from both train and test dataset since it is a redundant field.

## Exploratory Data Analysis (EDA):
Exploratory data analysis is performed to ensure that the data that is given to the model is valid and is compatible with the type of machine learning algorithm used in developing the model. It helps the model predict with greater accuracy.

### Data statistics:
Many statistic features can be derived from the features in the dataset in order to facilitate better understanding of the dataset to train the model perfectly. In the case of numerical attributes various statistics like mean, standard deviation, count etc. can be evaluated. These calculated statistics can be used by the machine learning model to extract useful information from the features and use it to form specific patterns or rules for predicting output
## Data Visualization: 
Some features can be better understood if they analyzed visually rather than dealing with pure numerical values. The redundant features or columns are removed from training and test dataset. The attack class is visualized, and the graph of attack classes is plotted

## DATA CORRELATION
A correlation coefficient is an approach to put worth to the relationship. Relationship coefficients have a worth of between - 1 and 1. A "0" signifies there is no connection between the factors by any stretch of the imagination, while - 1 or 1 implies that there is an ideal negative or positive relationship

## FEATURE SELECTION & DATA PREPARATION FOR MODELLING
All the features present in the dataset may not be necessary to predict the correct output. Some features can be removed from the dataset so that processing the dataset will become much easier and training the model will take less time. Feature selection solves this problem by analyzing the dataset thoroughly and with exploratory data analysis the user can select some of the many features from the dataset that contribute more to predict the output
feature selection algorithm identifies the features which are most important, and which have high contribution in predicting the output. These features are selected, and the model is instructed to give much more priority to these selected features.The plot helps to analyse easily, to select the best features.(heatmap1.png,heatmap2.png)
### Feature Selection for The Data:
After EDA, we have removed highly correlated features and selected Best Features using Chi Square Feature Selection

### Data Preparation for Modelling               
After Feature Selection, we have extracted the dataset with new features, applied Standardization to the data.  We have used Min Max scaler to Standardize the data. Data is being split into 67% and 33% for training and testing, respectively.  This prepared data is sent for Model Building     

## MODEL BUILDING, VALIDATION & EVALUATION
The Prepared Data is being used for Modelling. Here Data is Trained and Tested with various classifiers to filter out best Model which fits the data and predicts the attack behaviour according to our objective.
Classifiers Used:  Logistic, KNN, Decision Tree, SVM, Naïve Bayes.
testing_accuracy_models.png and training_accuracy_models.png gives the comparison between different models
Ensemble Classifiers Used: Adaboost, Random Forest, XG Boost Classifier.
testing_ensembles_models.png and training_ensembles_models.png gives the comparison between different ensemble models

The results are validated and evaluated from models_comparison.png,ensemble_models_comparison.png.
So from the observations these two png files,by observing the testing and training scores,precision and recall of all classifiers, Decision tree is efficient than other models to this dataset(we require good accuarcy ,high precision and recall,so area under the curve is high ).

## HYPERPARAMETER TUNING

hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned.

From Model Building, Validation & Evaluation, Decision Tree was the Chosen Classifier for the data.

From decision tree params we can see that max_depth is none which leads to overfitting ,gini is intended for continuous attributes and entropy is for atttributes that occur in class. So, Gini is the best choice.

After Hyperparameter Tuning, Best Parameters are obtained.  Now training & testing with Chosen Model and its Hyperparameters we got 99% accuracy.(model_test_results.txt from outputs)

Model is being trained with standardized data. To avoid disturbances, able to achieve the same results with high accuracy, we are using pipeline where in input data is standardized, sent for prediction.This pipeline is stored as pickle file 

After the pipeling is stored it is tested with random testing data which is in test.py which gave accurate results. The Objective of choosing Minimal Features and Predicting the type of attack with high accuracy is achieved


