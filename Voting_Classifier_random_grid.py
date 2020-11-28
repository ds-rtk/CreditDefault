
import pandas as pd
import numpy as np

train = pd.DataFrame(pd.read_csv('train.csv'))
test = pd.DataFrame(pd.read_csv('test.csv'))

train_raw = train.copy()
test_raw = test.copy()

from sklearn import preprocessing
# Create a label encoder object
le = preprocessing.LabelEncoder()
le_count = 0

# Iterate through the columns
for col in train_raw:
    if train_raw[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(train_raw[col].unique())) <= 2:
            # Train on the training data
            le.fit(train_raw[col])
            # Transform both training and testing data
            train_raw[col] = le.transform(train_raw[col])
            test_raw[col] = le.transform(test_raw[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
app_train = pd.get_dummies(train_raw)
app_test = pd.get_dummies(test_raw)

train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))



poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
poly_target = poly_features['TARGET']

poly_features = poly_features.drop(columns = ['TARGET'])

#Imputing missing value
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.fit_transform(poly_features_test)

#Polynomial transformation of degree 5

from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree = 5)


# Train polynomial features

poly_transformer.fit(poly_features)

# Transform the feature
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)

print('Poly train shape : ', poly_features.shape)
print('Poly test shape : ', poly_features_test.shape)


# Create a dataframe of the features 
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Put test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))
# Merge polynomial features into training dataframe
drop_columns = list(set(list(app_train.columns)).intersection(list(poly_features.columns)))
poly_features.drop(drop_columns, inplace=True, axis=1)

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']

app_train_poly = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# Merge polnomial features into testing dataframe

drop_columns_test = list(set(list(app_test.columns)).intersection(list(poly_features_test.columns)))
poly_features_test.drop(drop_columns_test, inplace=True, axis=1)

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']

app_test_poly = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape:  ', app_test_poly.shape)



app_train_domain = app_train_poly.copy()
app_test_domain = app_test_poly.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']



numeric_summary_table_train = pd.DataFrame(app_train_domain[list(app_train_domain.select_dtypes(['int64','float64']).columns)].describe().transpose())
numeric_summary_table_train['colnames'] = app_train_domain.select_dtypes(['int64','float64']).columns



numeric_summary_table_train_maxgt1 = numeric_summary_table_train[numeric_summary_table_train['max'] > 1]
std_columns_list = list(numeric_summary_table_train_maxgt1['colnames'])
columns_not_considered = ['SK_ID_CURR','CNT_CHILDREN']
std_columns_list = list(set(std_columns_list).difference(set(columns_not_considered)))


train_mod = app_train_domain.copy()
test_mod = app_test_domain.copy()
print (train_mod.shape)
print (test_mod.shape)


train_labels = train_mod['TARGET']
# Feature names
train_mod = train_mod.drop('TARGET',axis=1)
features = list(train_mod.columns)


from sklearn.preprocessing import MinMaxScaler, Imputer

# Median imputation of missing values
imputer = Imputer(strategy = 'median')


# Fit on the training data
imputer.fit(train_mod)

# Transform both training and testing data
train_mod = imputer.transform(train_mod)
test_mod = imputer.transform(test_mod)


# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))


# Repeat with the scaler
scaler.fit(train_mod)
train_mod = scaler.transform(train_mod)
test_mod = scaler.transform(test_mod)



print('Final train shape: ',train_mod.shape)
print('Final test shape: ',test_mod.shape)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')

print ('Model run start')

voting_clf.fit(train_mod, train_labels)

print ('Model run complete')

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(train_mod, train_labels)
    train_pred = clf.predict(train_mod)
    print(clf.__class__.__name__, accuracy_score(train_labels,train_pred))

print ('Prediction dataset run Start')

test_pred = pd.DataFrame(voting_clf.predict(test_mod))
test_pred.to_csv('test_pred.csv')

print ('Final test prediction csv generated')

