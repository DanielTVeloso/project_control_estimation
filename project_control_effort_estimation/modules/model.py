from math import log
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time

from category_encoders import CatBoostEncoder
from category_encoders import LeaveOneOutEncoder
from category_encoders import TargetEncoder
from category_encoders import BinaryEncoder 
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler # data normalization with sklearn
from sklearn.preprocessing import StandardScaler # data standardizaiton with sklearn
from sklearn.preprocessing import PowerTransformer # data transformation(box-cox,...) with sklearn
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import joblib
from sklearn.preprocessing import OneHotEncoder


database_name = "proj_ctrldb"
database_user = "root"

feature_list = []

#Server=localhost;Database=master;Trusted_Connection=True;

def DBconnection():
    try:
        con = pymysql.connect(database=database_name, user=database_user, password="root", host="localhost", port=3306)
        cur = con.cursor()
        cur.execute("select t.id, (t.finish - t.start + 1) as task_duration, te_view.expected_effort as task_effort, t.number as task_number,p.aicos_funding_rate as afr, p.equipment_funding_type_mask as eftm, p.backlog_type_mask as btm, p.revenue_distribution_mask as rdm, p.type_mask as p_type_mask, p.department_mask, p.business_field_mask, p.business_subfield_mask, p.priority_mask, b.status_mask as b_stats, b.funding_total as b_funds, b.budget_indirect_model_mask as bimm, b.funding_total_costs as b_funds_cost, b.aicos_partner_number as apn, b.overhead_funding_type_mask as ofm, b.overhead_funding_rate as ofr, b.duration as b_duration, b.type_mask as b_type_mask, b.cost_type_mask, b.stage_mask as b_stage_mask, w.number as wp_number, w.type_mask as wp_type_mask, (w.finish - w.start + 1) as wp_duration from projects p join baselines b on p.id = b.project_id join work_packages w on b.id = w.baseline_id join tasks t on w.id = t.work_package_id join task_effort_view te_view on te_view.task_id = t.id;")
        tasks = cur.fetchall()
        #print(tasks)
        
    except (Exception, pymysql.Error) as error :
        print ("Error while fetching data from MySQL", error)

    finally:
    #closing database connection.
        if(con):
            cur.close()
            con.close()
            print("MySQL connection is closed")

    # transform db data to pandas dataframe
    df_tasks = pd.DataFrame(tasks, columns=['id','task_duration', 'task_effort', 'task_number', 'afr', 'eftm', 'btm', 'rdm', 'p_type', 'department', 'business_field', 'business_subfield', 'priority', 'b_stats', 'b_funds', 'bimm', 'b_funds_cost', 'apn', 'ofm', 'ofr', 'b_duration', 'b_type', 'b_cost_type', 'b_stage', 'wp_number', 'wp_type', 'wp_duration'])
    df_tasks.fillna(4.0, inplace = True)
    df_tasks['b_funds'] = pd.to_numeric(df_tasks['b_funds'], downcast="float")
    df_tasks['b_funds_cost'] = pd.to_numeric(df_tasks['b_funds_cost'], downcast="float")
    #df_tasks = df_tasks.drop(['id','bimm','b_stage','ofm','b_stats','b_type'], axis=1)
    #df_tasks = df_tasks.drop(['priority','ofr','business_field','b_funds_cost','btm','afr','bimm','b_stage','ofm','b_stats','b_type'], axis=1)
    # For TASK_DURATION
    df_tasks = df_tasks.drop(['b_funds_cost','btm','afr','bimm','b_stage','b_stats','business_field','ofr','ofm','b_type'], axis = 1)
    # For TASK_EFFORT
    #df_tasks = df_tasks.drop(['b_stage','bimm','ofm','priority','btm','b_funds_cost','afr','apn','b_stats','b_type'], axis = 1)
    # get the labels and merge it with database data
    df_labels = pd.read_csv('output.csv')

    # print(df_labels.info())  # tells us there is approx. 4000 missing values
    # impute the missing values by a generative string 'others'
    #df_labels.dropna(inplace=True)
    df_labels['labels'].fillna('Others', inplace = True)
    #df_labels['labels'] = df_labels['labels'].apply(lambda x: filter_labels(x))
    df_tasks = pd.merge(df_tasks, df_labels, on='id', how='inner').drop(['id'], axis = 1)

    # cut the effort error samples
    for x in df_tasks.index:
        if df_tasks.loc[x, 'task_effort'] > 100:
            df_tasks.drop(x, inplace = True)


    #df_tasks = label_encoding_labels(df_labels,df_tasks)
    #df_tasks = binary_encoding_labels(df_labels,df_tasks)
    #df_tasks = one_hot_encoding_labels(df_labels,df_tasks)
    #df_tasks = target_encoding_labels(df_labels,df_tasks)
    #df_tasks = cat_boost_encoding_labels(df_labels,df_tasks)
    #df_tasks = leave_one_out_encoding_labels(df_labels,df_tasks)

    '''
    df_tasks_tmp = df_tasks.drop(['task_duration','task_effort','labels'], axis = 1)
    df_tasks = df_tasks[['id','task_duration','task_effort','labels']]
    Q1 = df_tasks_tmp.quantile(0.25)
    Q3 = df_tasks_tmp.quantile(0.75)
    IQR = Q3 - Q1
    df_tasks_tmp = df_tasks_tmp[~((df_tasks_tmp < (Q1 - 1.5 * IQR)) |(df_tasks_tmp > (Q3 + 1.5 * IQR))).any(axis=1)]

    df_tasks = pd.merge(df_tasks, df_tasks_tmp, on='id', how='inner')
    df_tasks = df_tasks.drop(['id'], axis = 1)

    Q1 = df_tasks.quantile(0.25)
    Q3 = df_tasks.quantile(0.75)
    IQR = Q3 - Q1
    df_tasks = df_tasks[~((df_tasks < (Q1 - 1.5 * IQR)) |(df_tasks > (Q3 + 1.5 * IQR))).any(axis=1)]
    df_tasks = df_tasks.drop(['department','wp_type'], axis = 1)
    '''
    df_tasks_norm, df_tasks_stand = df_tasks.copy(), df_tasks.copy()
    for norm_f in df_tasks.columns.tolist():
        if norm_f not in ['task_duration','task_effort','labels']:
            df_tasks[norm_f] = df_tasks[norm_f]
            #df_tasks[norm_f] = PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(np.array(df_tasks[norm_f].apply(lambda x: x+1.0)).reshape(-1,1))
            df_tasks_stand[norm_f] = StandardScaler().fit_transform(np.array(df_tasks_stand[norm_f].apply(float)).reshape(-1,1))
            #df_tasks[norm_f] = PowerTransformer(method='box-cox', standardize=True).fit_transform(np.array(df_tasks[norm_f].apply(lambda x: x+1.0)).reshape(-1,1))
            df_tasks_norm[norm_f] = MinMaxScaler().fit_transform(np.array(df_tasks_norm[norm_f].apply(float)).reshape(-1,1))
        else:
            df_tasks[norm_f] = df_tasks[norm_f]
            df_tasks_stand[norm_f] = df_tasks_stand[norm_f]
            df_tasks_norm[norm_f] = df_tasks_norm[norm_f]


    #print(df_tasks.head())
    print(df_tasks.describe())
    #print(df_tasks.corr())
    #correlationMatrix(df_tasks)
    #dataAnalysisGraphs(df_tasks)
    
    encoding = target_encoding_labels
    print("Final Ensemble\n\n")
    finalEnsemble(df_tasks,df_tasks_stand,encoding)
    '''
    print("Random Forest\n\n")
    RandomForestDeliveryForecast(df_tasks,encoding)
    RandomForestDeliveryForecast(df_tasks_stand,encoding)
    RandomForestDeliveryForecast(df_tasks_norm,encoding)
    print("Extra Tree\n\n")
    ExtraTreeDeliveryForecast(df_tasks,encoding)
    ExtraTreeDeliveryForecast(df_tasks_stand,encoding)
    ExtraTreeDeliveryForecast(df_tasks_norm,encoding)
    print("XGBoost\n\n")
    XGBoostDeliveryForecast(df_tasks,encoding)
    XGBoostDeliveryForecast(df_tasks_stand,encoding)
    XGBoostDeliveryForecast(df_tasks_norm,encoding)
    print("Gradient Boosting Tree\n\n")
    BstTreeDeliveryForecast(df_tasks,encoding)
    BstTreeDeliveryForecast(df_tasks_stand,encoding)
    BstTreeDeliveryForecast(df_tasks_norm,encoding)
    print("Lasso\n\n")
    LassoDeliveryForecast(df_tasks,encoding)
    LassoDeliveryForecast(df_tasks_stand,encoding)
    LassoDeliveryForecast(df_tasks_norm,encoding)
    print("KNN\n\n")
    KNNDeliveryForecast(df_tasks,encoding)
    KNNDeliveryForecast(df_tasks_stand,encoding)
    KNNDeliveryForecast(df_tasks_norm,encoding)
    print("SVR\n\n")
    SVRDeliveryForecast(df_tasks,encoding)
    SVRDeliveryForecast(df_tasks_stand,encoding)
    SVRDeliveryForecast(df_tasks_norm,encoding)
    print("MLP-ANN\n\n")
    MLPDeliveryForecast(df_tasks,encoding)
    MLPDeliveryForecast(df_tasks_stand,encoding)
    MLPDeliveryForecast(df_tasks_norm,encoding)
    '''

def filter_labels(l):
    if len(l.rsplit(";")) > 1:
        return l.rsplit(";")[0]
    else:
        return l

def remove_outliers(train_X, train_y):
    train = pd.concat([train_X, train_y], axis = 1)
    t = train[['task_number','wp_number','wp_duration','b_duration']]
    Q1 = t.quantile(0.25)
    Q3 = t.quantile(0.75)
    IQR = Q3 - Q1
    t = t[~((t < (Q1 - 1.5 * IQR)) |(t > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    df_outlier_removed=pd.DataFrame(t)
    ind_diff=train.index.difference(df_outlier_removed.index)

    for i in range(0, len(ind_diff),1):
        train=train.drop([ind_diff[i]])

    return train.drop(['task_duration'], axis = 1), train['task_duration']

def train_save_model(model, train, encoding, model_name):
    """
    Train model especified using all training data

    Returns: Can save trained model on a file if save_path is specified
    """ 
    train_y = train['task_duration']
    train_X = train.drop(['task_duration','task_effort'],axis=1)
    train_X.head().to_csv('X_train.csv', encoding='utf-8', index = False)
    if encoding != None:
        train_X, test_X = encoding(train_X, train_y, train_X, save=True)
        train_X.head().to_csv('X_train_encoded.csv', encoding='utf-8', index = False)
    model.fit(train_X, train_y)
    joblib.dump(model, '../storage/models/'+model_name+'_trained.joblib') #save trained_model to disk
    
def cv_score(model, train, cv, encoding, model_name):

    train_labels = train['task_duration']
    train_features = train.drop(['task_duration','task_effort'],axis=1)
    
    fold_splits = KFold(n_splits=cv).split(train_features, train_labels)
    i = 1
    mae_scores, rmse_scores, r2_scores, mmre_scores, pred_scores, fit_time, fold_models, predicts, t_y = [], [], [], [], [], [], [], [], []
    train_X = None

    for train_ind, test_ind in fold_splits:
        print('Started fold {}/2'.format(i))

        train_X, test_X = train_features.iloc[train_ind], train_features.iloc[test_ind]
        train_y, test_y = train_labels.iloc[train_ind], train_labels.iloc[test_ind]
        #train_X, train_y = remove_outliers(train_X, train_y)
        train_X.head().to_csv('X_train.csv', encoding='utf-8', index = False)
        if encoding != None:
            train_X, test_X = encoding(train_X,train_y,test_X)
        
        start = time.time()
#        #print(train_X.head())
        train_X.head().to_csv('X_train_encoded.csv', encoding='utf-8', index = False)   
        model.fit(train_X, train_y)
        stop = time.time()
        # Use the forest's predict method on the test data
        predictions = model.predict(test_X)
        # Calculate the absolute errors
        errors = abs(predictions - np.array(test_y))
        mae_scores.append(round(np.mean(errors), 2))
        rmse_scores.append(round(mean_squared_error(test_y, predictions, squared=False), 2))
        r2_scores.append(round(r2_score(test_y,predictions), 2))
        mmre_scores.append(round(np.mean(np.divide(errors,test_y)), 2))
        # Calculate PRED(0.3)
        pred = []
        for x in np.divide(errors,test_y):
            if x <= 0.3:
                pred.append(1.0)
            else: pred.append(0.0)
        pred_scores.append(round(np.mean(pred), 2))
        fit_time.append(round(stop - start,2))
        fold_models.append(model)
        predicts.append(predictions)
        t_y.append(test_y)
        i += 1
    
    train_X = train_X.rename(columns={'labels': 'task_type'})
    global feature_list

    feature_list = train_X.columns

    return { 'test_mae': np.array(mae_scores),
    'test_rmse': np.array(rmse_scores),
    'test_r2': np.array(r2_scores),
    'test_mmre': np.array(mmre_scores),
    'test_pred': np.array(pred_scores),
    'fit_time': np.array(fit_time),
    'models': fold_models,
    'predictions': predicts,
    'test_y': t_y }

def finalEnsemble(df_tasks,df_tasks_stand,encoding):
    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    train_stand, val_stand = train_test_split(df_tasks_stand, test_size = 0.25, random_state = 42)
    rf = RandomForestRegressor(max_depth=60,max_features=5,min_samples_leaf=1,min_samples_split=2, n_estimators = 600, random_state=42)
    #rf = RandomForestRegressor(max_depth=100,max_features='log2',min_samples_leaf=1,min_samples_split=2, n_estimators = 800, random_state=42)
    extra = ExtraTreesRegressor(max_depth=80,max_features='auto',min_samples_leaf=1,min_samples_split=2, n_estimators = 600, random_state=42)
    #extra = ExtraTreesRegressor(max_depth=60,max_features='auto',min_samples_leaf=1,min_samples_split=2, n_estimators = 100, random_state=42)
    xgb = XGBRegressor(colsample_bytree=0.7, gamma=0.0, learning_rate=0.1, max_depth=40, min_child_weight = 1, subsample=1.0, random_state=42)
    bstTree = GradientBoostingRegressor(learning_rate=0.01,max_depth=20,max_features=5,min_samples_leaf=1,min_samples_split=5, n_estimators = 900, random_state=42)
    #bstTree = GradientBoostingRegressor(learning_rate=0.01,max_depth=20,max_features='sqrt',min_samples_leaf=1,min_samples_split=5, n_estimators = 800, random_state=42)
    neigh = KNeighborsRegressor(n_neighbors=3)
    svr = SVR(C=100, gamma=0.1, kernel='rbf', cache_size=7000)
    mlp = MLPRegressor(activation='tanh', hidden_layer_sizes=(50, 100, 150), solver='adam', learning_rate='constant', alpha=0.001, max_iter=2000, random_state = 42)
    #mlp = MLPRegressor(activation='tanh', hidden_layer_sizes=(150, 50, 100), solver='adam', learning_rate='constant', alpha=0.0001, max_iter=2000, random_state = 42)

    train_save_model(rf, train, encoding, 'random_forest')
    rf = cv_score(rf, train, 2, encoding, 'random_forest')
    predictedVSactualAnalyse(rf['test_y'][0], rf['predictions'][0])

    train_save_model(extra, train, encoding, 'extratrees')
    extra = cv_score(extra, train, 2, encoding, 'extratrees')['predictions']

    train_save_model(xgb, train, encoding, 'xboost') 
    xgb = cv_score(xgb, train, 2, encoding, 'xboost')['predictions']
    
    train_save_model(bstTree, train, encoding, 'gbm')
    bstTree = cv_score(bstTree, train, 2, encoding, 'gbm')['predictions']

    train_save_model(neigh, train, encoding, 'knn')
    neigh = cv_score(neigh, train, 2, encoding, 'knn')['predictions'] 

    #train_save_model(svr, train, encoding, 'svr')
    #svr = cv_score(svr, train, 2, encoding, 'svr')['predictions']
    
    train_save_model(mlp, train, encoding, 'mlp')
    mlp = cv_score(mlp, train_stand, 2, encoding, 'mlp')['predictions']

    val = input("Enter your value after saving models: ")

    predictions = rf['predictions']
    for i in range(len(extra)):
        predictions[i] = np.add(predictions[i],extra[i])
    for i in range(len(xgb)):
        predictions[i] = np.add(predictions[i],xgb[i])
    for i in range(len(bstTree)):
        predictions[i] = np.add(predictions[i],bstTree[i])
    for i in range(len(neigh)):
        predictions[i] = np.add(predictions[i],neigh[i])
    for i in range(len(svr)):
        predictions[i] = np.add(predictions[i],svr[i])
    for i in range(len(mlp)):
        predictions[i] = np.add(predictions[i],mlp[i])

    for i in range(len(predictions)):
        predictions[i] = predictions[i] / 7

    mae_scores, rmse_scores, r2_scores, mmre_scores, pred_scores = [], [], [], [], []
    for p, t in zip(predictions,rf['test_y']):
        # Calculate the absolute errors
        errors = abs(p - np.array(t))
        mae_scores.append(round(np.mean(errors), 2))
        rmse_scores.append(round(mean_squared_error(t, p, squared=False), 2))
        r2_scores.append(round(r2_score(t,p), 2))
        mmre_scores.append(round(np.mean(np.divide(errors,t)), 2))
        # Calculate PRED(0.3)
        pred = []
        for x in np.divide(errors,t):
            if x <= 0.3:
                pred.append(1.0)
            else: pred.append(0.0)
        pred_scores.append(round(np.mean(pred), 2))

    predictedVSactualAnalyse(rf['test_y'][0], predictions[0])

    print('Mean Absolute Error:', abs(round(np.array(mae_scores).mean(),2)))
    print('Root Mean Squared Error:', abs(round(np.array(rmse_scores).mean(),2)))
    print('R2:', round(np.array(r2_scores).mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(np.array(mmre_scores).mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(np.array(pred_scores).mean(),2)))

def RandomForestDeliveryForecast(df_tasks, encoding):

    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

    max_features = ['sqrt','log2','auto'] #livro de khun recomenda 10 valores equally even entre 2-nr de predictors
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2] #[2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1] #[1, 4]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}

    rf = RandomForestRegressor()
    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)

    rf = parameterTuning(rf,param_grid,'neg_root_mean_squared_error',5,val_features,val_labels)
    '''
    #max_features='sqrt' for TASK_DURATION
    #rf = RandomForestRegressor(max_depth=60,max_features=5,min_samples_leaf=1,min_samples_split=2, n_estimators = 600, random_state=42)
    #max_features= 'log2' for TASK_EFFORT
    rf = RandomForestRegressor(max_depth=100,max_features='log2',min_samples_leaf=1,min_samples_split=2, n_estimators = 800, random_state=42)
    scores = cv_score(rf, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

    rf = scores['models'][0]
    #print(rf.get_params())
    #featureSelectionGraph(rf)

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    #cumulativeImpGraph(importances, feature_importances)

    '''
    # Plot the actual values
    print(predictions, test_labels)
    
    #plt.plot([x for x in range(len(test_labels))], test_labels, 'b-', label = 'actual')
    # Plot the predicted values
    #plt.plot([x for x in range(len(test_labels))], predictions, 'ro', label = 'prediction')
    #plt.legend()
    
    fig, axs = plt.subplots(nrows=1,ncols=2,sharey=True)
    axs[0].hist(predictions, bins=35, density=True, stacked=True, color='lightblue')
    axs[1].hist(test_labels, bins=35, density=True, stacked=True, color='red')

    # Set common labels
    fig.suptitle('Predicted and Actual Task Durations Distribution')
    fig.text(0.5, 0.04, 'Task Duration', ha='center', va='center')
    fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')

    # Graph labels
    plt.xlabel('Task_Duration')
    plt.ylabel('Frequency')
    plt.title('Actual and Predicted task duration distribution')


    plt.scatter(predictions, test_labels)
    plt.show()
    '''

def ExtraTreeDeliveryForecast(df_tasks, encoding):

    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

    max_features = ['sqrt','log2','auto']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2] #[2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1] #[1, 4]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}

    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)

    extra = parameterTuning(ExtraTreesRegressor(),param_grid,'neg_root_mean_squared_error',5,val_features,val_labels)
    '''
    # for TASK_DURATION 
    #extra = ExtraTreesRegressor(max_depth=80,max_features='auto',min_samples_leaf=1,min_samples_split=2, n_estimators = 600, random_state=42)
    # for TASK_EFFORT 
    extra = ExtraTreesRegressor(max_depth=60,max_features='auto',min_samples_leaf=1,min_samples_split=2, n_estimators = 100, random_state=42)

    scores = cv_score(extra, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

def XGBoostDeliveryForecast(df_tasks, encoding):

    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    param_grid = {"learning_rate": [0.001, 0.01, 0.1, 0.2], #[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        "max_depth"        : [20, 40, 60], #80,100, None
        "min_child_weight" : [ 1, 4, 7 ],
        "subsample"        : [ 0.6, 0.8, 1.0],
        "gamma"            : [ 0.0, 0.1, 0.2], #, 0.3, 0.4 
        "colsample_bytree" : [ 0.5, 0.7, 0.9 ] 
    }
    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)

    xgb = parameterTuning(XGBRegressor(),param_grid,'neg_root_mean_squared_error',3,val_features,val_labels)
    '''
    xgb = XGBRegressor(colsample_bytree=0.7, gamma=0.0, learning_rate=0.1, max_depth=40, min_child_weight = 1, subsample=1.0, random_state=42)

    scores = cv_score(xgb, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

def BstTreeDeliveryForecast(df_tasks, encoding):

    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    #"learning_rate": [0.15,0.1,0.05,0.01,0.005,0.001]
    param_grid = {"learning_rate": [0.2,0.1,0.01,0.001,0.0001],
        "n_estimators"     : [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
        "max_depth"        : [20, 40, 60, 80, 100, None],
        "max_features"     : ['sqrt','log2','auto'],
        "min_samples_split": [2, 5], #[2, 5, 10, 60, 100]
        "min_samples_leaf" : [1] #[1, 4, 9]
    }
    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)
    
    bstTree = parameterTuning(GradientBoostingRegressor(),param_grid,'neg_root_mean_squared_error',5,val_features,val_labels)
    '''
    #'max_features': 'sqrt' for TASK_DURATION
    #bstTree = GradientBoostingRegressor(learning_rate=0.01,max_depth=20,max_features=5,min_samples_leaf=1,min_samples_split=5, n_estimators = 900, random_state=42)
    #for TASK_EFFORT
    bstTree = GradientBoostingRegressor(learning_rate=0.01,max_depth=20,max_features='sqrt',min_samples_leaf=1,min_samples_split=5, n_estimators = 800, random_state=42)

    scores = cv_score(bstTree, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

def LassoDeliveryForecast(df_tasks, encoding):

    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    param_grid = dict()
    param_grid['alpha'] = np.arange(0, 1, 0.01)
    # for duration
    #lasso = Lasso(alpha=0.01)
    lasso = Lasso(alpha=0.1)
    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)

    lasso = parameterTuning(Lasso(),param_grid,'neg_root_mean_squared_error',5,val_features,val_labels)
    '''
    scores = cv_score(lasso, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

    lasso = scores['models'][0]
    #print(lasso.get_params())

    # Get numerical feature importances
    importances = list(lasso.coef_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

def KNNDeliveryForecast(df_tasks, encoding):

    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    param_grid = dict()
    param_grid['n_neighbors'] = np.arange(3, 10, 2)
    
    neigh = KNeighborsRegressor(n_neighbors=3)
    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)

    neigh = parameterTuning(KNeighborsRegressor(),param_grid,'neg_root_mean_squared_error',5,val_features,val_labels)
    '''

    scores = cv_score(neigh, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

def SVRDeliveryForecast(df_tasks, encoding):
    
    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    param_grid = {'C': [0.001,0.1,1, 10, 100,1000], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'sigmoid']} #'poly'
    
    svr = SVR(C=100, gamma=0.1, kernel='rbf', cache_size=7000)
    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)

    svr = parameterTuning(SVR(),param_grid,'neg_root_mean_squared_error',5,val_features,val_labels)
    '''
    scores = cv_score(svr, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

def MLPDeliveryForecast(df_tasks, encoding):

    # Split the data into training and validation(tuning) sets
    train, val = train_test_split(df_tasks, test_size = 0.25, random_state = 42)
    
    # parameter tuning
    param_grid = {'activation': ['relu', 'tanh', 'logistic'],
          'hidden_layer_sizes': list(itertools.permutations([50,100,150],2)) + list(itertools.permutations([50,100,150],3)) + [50,100,150],
          #'solver': ['adam', 'lbfgs'],
          #'learning_rate' : ['constant', 'adaptive'],
          'alpha': [0.0001,0.001,0.01,0.1]
        }
    #for task_duration
    #mlp = MLPRegressor(activation='tanh', hidden_layer_sizes=(50, 100, 150), solver='adam', learning_rate='constant', alpha=0.001, max_iter=2000, random_state = 42)
    
    mlp = MLPRegressor(activation='tanh', hidden_layer_sizes=(150, 50, 100), solver='adam', learning_rate='constant', alpha=0.0001, max_iter=2000, random_state = 42)
    
    '''
    val = encoding(val)
    val_labels = np.array(val['task_effort'])
    val_features = val.drop(['task_effort'],axis=1)
    val_features = np.array(val_features)

    mlp = parameterTuning(MLPRegressor(max_iter=2000),param_grid,'neg_root_mean_squared_error',5,val_features,val_labels)
    '''
    scores = cv_score(mlp, train, 5, encoding)

    print('Mean Absolute Error:', abs(round(scores['test_mae'].mean(),2)))
    print('Root Mean Squared Error:', abs(round(scores['test_rmse'].mean(),2)))
    print('R2:', round(scores['test_r2'].mean(),2))
    print('Mean Magnitude of Relative Error:', abs(round(scores['test_mmre'].mean(),2)))
    print('Percentage Relative Error Deviation (PRED(0.3)):', abs(round(scores['test_pred'].mean(),2)))
    print('Mean Train Time:', round(scores['fit_time'].mean(),2))

def parameterTuning(model,param_grid,scoring,cv,features,labels):

    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                           scoring = scoring, cv = cv, 
                           n_jobs = -1, verbose = 2)
    
    grid_search.fit(features, labels)

    print(grid_search.best_params_)

    return grid_search.best_estimator_

def label_encoding_labels(train_X, train_y = None, test_X = None, save=False):

    encoder = LabelEncoder()
    train_X['labels'] = encoder.fit_transform(train_X['labels'])
    if type(train_y) == type(None) and type(test_X) == type(None):
        return train_X
    else:
        if save:
            joblib.dump(encoder, '../storage/models/encoder_trained.joblib') #save trained_model to disk
        test_X['labels'] = encoder.fit_transform(test_X['labels'])
        return train_X, test_X

def one_hot_encoding_labels(train_X, train_y = None, test_X = None):
    train_X = pd.get_dummies(train_X, columns=["labels"])
    if type(train_y) == type(None) and type(test_X) == type(None):
        return train_X
    else:
        test_X = pd.get_dummies(test_X, columns=["labels"])
        return train_X, test_X

def binary_encoding_labels(train_X, train_y = None, test_X = None):

    encoder = BinaryEncoder(cols = ['labels'])  
    lables_encoded = encoder.fit_transform(train_X['labels'])
    # concat the final df with the categorical label column
    train_X = pd.concat([train_X, lables_encoded], axis=1).drop(['labels'],axis=1)

    if type(train_y) == type(None) and type(test_X) == type(None):
        return train_X
    else:
        lables_encoded = encoder.fit_transform(test_X['labels'])
        test_X = pd.concat([test_X, lables_encoded], axis=1).drop(['labels'],axis=1)
        return train_X, test_X

def target_encoding_labels(train_X, train_y = None, test_X = None, save = False):

    if type(train_y) == type(None) and type(test_X) == type(None):
        TargetEnc = TargetEncoder(handle_missing='return_nan')
        values = TargetEnc.fit_transform(X = train_X['labels'], y = train_X['task_effort'])


        train_X = train_X.drop(['labels'], axis = 1)  
        train_X = pd.concat([train_X, values], axis = 1)
        return train_X
    else:
        TargetEnc = TargetEncoder(handle_missing='return_nan')
        values = TargetEnc.fit_transform(X = train_X['labels'], y = train_y)
        if save:
            joblib.dump(TargetEnc, '../storage/models/TargetEnc_trained.joblib') #save trained_model to disk
        test_values = TargetEnc.transform(test_X['labels'])
        '''
        print(values.info())
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
        values['labels'] = imputer.fit_transform(np.array(values['labels']).reshape(-1,1))
        test_values['labels'] = imputer.fit_transform(np.array(test_values['labels']).reshape(-1,1))
        print(values.info())
        '''
        train_X = train_X.drop(['labels'], axis = 1)  
        train_X = pd.concat([train_X, values], axis = 1)
        test_X = test_X.drop(['labels'], axis = 1)  
        test_X = pd.concat([test_X, test_values], axis = 1)

        return train_X, test_X

def cat_boost_encoding_labels(train_X, train_y = None, test_X = None):

    if type(train_y) == type(None) and type(test_X) == type(None):
        CatBoostEnc = CatBoostEncoder()
        values = CatBoostEnc.fit_transform(X = train_X['labels'], y = train_X['task_duration'])

        train_X = train_X.drop(['labels'], axis = 1)  
        train_X = pd.concat([train_X, values], axis = 1)
        return train_X
    else:
        CatBoostEnc = CatBoostEncoder()
        values = CatBoostEnc.fit_transform(X = train_X['labels'], y = train_y)
        test_values = CatBoostEnc.transform(test_X['labels'])

        train_X = train_X.drop(['labels'], axis = 1)  
        train_X = pd.concat([train_X, values], axis = 1)
        test_X = test_X.drop(['labels'], axis = 1)  
        test_X = pd.concat([test_X, test_values], axis = 1)

        return train_X, test_X

def leave_one_out_encoding_labels(train_X, train_y = None, test_X = None):

    if type(train_y) == type(None) and type(test_X) == type(None):
        LeaveOneOutEnc = LeaveOneOutEncoder()
        values = LeaveOneOutEnc.fit_transform(X = train_X['labels'], y = train_X['task_duration'])

        train_X = train_X.drop(['labels'], axis = 1)  
        train_X = pd.concat([train_X, values], axis = 1)
        return train_X
    else:
        LeaveOneOutEnc = LeaveOneOutEncoder()
        values = LeaveOneOutEnc.fit_transform(X = train_X['labels'], y = train_y)
        test_values = LeaveOneOutEnc.transform(test_X['labels'])

        train_X = train_X.drop(['labels'], axis = 1)  
        train_X = pd.concat([train_X, values], axis = 1)
        test_X = test_X.drop(['labels'], axis = 1)  
        test_X = pd.concat([test_X, test_values], axis = 1)

        return train_X, test_X

def dataAnalysisGraphs(df_tasks):

    #----ESTUDO DISTRIBUIÇÕES----
    '''
    fig, axs = plt.subplots(2, 8, figsize=(7, 7))
    
    aux = 0
    for col in df_tasks.columns:
        sns.histplot(data=df_tasks[col], kde=True, stat='probability', ax=axs[aux//8, aux%8])
        axs[aux//8, aux%8].set_ylabel('')
        aux += 1
    '''
    #----ESTUDO DOS PLOT BOXES DE TODOS AS VARS----
    '''
    fig, axs = plt.subplots(2, 5, figsize=(7, 5),sharey=False)
    #axs.set(ylabel='probability')
    axs[0,1].set_ylabel(' ')
    axs[0,2].set_ylabel(' ')
    axs[0,3].set_ylabel(' ')
    axs[0,4].set_ylabel(' ')
    axs[1,1].set_ylabel(' ')
    axs[1,2].set_ylabel(' ')
    axs[1,3].set_ylabel(' ')
    
    aux = 0
    for col in df_tasks.columns:
        sns.histplot(data=df_tasks[col], kde=True, stat='probability', ax=axs[aux//5, aux%5],)
        axs[aux//5, aux%5].set_xlabel(col)
        aux += 1
    '''
    #----ESTUDO RELAÇÕES VARS IND COM DEP----
    sns.pairplot(df_tasks, x_vars=df_tasks.drop(['task_duration','task_effort'],axis=1).columns, y_vars=["task_duration", "task_effort"])

    #sns.histplot(data=df_tasks, x="task_effort",stat="probability",bins=40)

    plt.show()

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df_labels.groupby('labels').agg(['count']).stack())

def predictedVSactualAnalyse(actuals, predictions):
    '''
    plt.plot([x for x in range(len(actuals))], actuals, 'b-', label = 'actual')
    # Plot the predicted values
    plt.plot([x for x in range(len(actuals))], predictions, 'ro', label = 'prediction')
    plt.legend()
    '''
    lineStart = actuals.min() 
    lineEnd = actuals.max()  
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')

    # Graph labels
    plt.xlabel('prediction')
    plt.ylabel('actual')
    #plt.title('Actual and Predicted task duration distribution')

    plt.scatter(predictions, actuals)
    plt.show()

def featureSelectionGraph(rf):
    importances = rf.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in rf.estimators_], axis=0)

    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 3), stdev) for feature, importance, stdev in zip(feature_list, importances, std)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    features = [f_i[0] for f_i in feature_importances]
    importances = [f_i[1] for f_i in feature_importances]
    std = [f_i[2] for f_i in feature_importances]

    forest_importances = pd.Series(importances, index=features)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    plt.show()

def cumulativeImpGraph(importances, feature_importances):
    # list of x locations for plotting
    x_values = list(range(len(importances)))

    # List of features sorted from most to least important
    sorted_importances = [importance[1] for importance in feature_importances]
    sorted_features = [importance[0] for importance in feature_importances]
    # Cumulative importances
    cumulative_importances = np.cumsum(sorted_importances)
    # Make a line graph
    plt.plot(x_values, cumulative_importances, 'g-')
    # Draw line at 95% of importance retained
    plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
    # Draw vertical line at 95% of importance retained
    plt.vlines(x = 16, ymin=0.25, ymax=1.0, color = 'b', linestyles = 'solid')
    # Format x ticks and labels
    plt.xticks(x_values, sorted_features, rotation = 'vertical')
    # Axis labels and title
    plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances')
    plt.show()

def correlationMatrix(df_tasks):
    corr = df_tasks.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        annot=True,
        fmt=".2f"
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()

def predict_from_model(request_data, load_path=''):
    """Predict results using the previously trained models (random_forest as default) and
    the data received in the request
    
    Parameters:
        request_data(json): Data from POST request
        load_path(str): if load path is specified, load storaged models from that path
    
    Returns:
        results(json): Results of the predictions
        status_code(int): HTTP status code to return
    """
    # check request data for errors before proceeding
    #has_error, error_messages = utils.check_request_data(request_data)
    #if has_error:
    #    return {
    #        'error': has_error,
    #        'messages': error_messages
    #    }, 400
    if load_path == '':
        load_path = '../storage/models'
    results = []
    for i in range(len(request_data)):
        numerical_features = {
            "task_number" : [request_data[i]['task_number']],
            "eftm": [request_data[i]['eftm']],
            "rdm" : [request_data[i]['rdm']],
            "p_type" : [request_data[i]['p_type']],
            "department" : [request_data[i]['department']],
            "business_subfield": [request_data[i]['business_subfield']],
            "priority" : [request_data[i]['priority']],
            "b_funds" : [request_data[i]['b_funds']],
            "apn": [request_data[i]['apn']],
            "b_duration" : [request_data[i]['b_duration']],
            "b_cost_type" : [request_data[i]['b_cost_type']],
            "wp_number" : [request_data[i]['wp_number']],
            "wp_type" : [request_data[i]['wp_type']],
            "wp_duration": [request_data[i]['wp_duration']],
        }
        df_numerical = pd.DataFrame(numerical_features)

        labels = {
            "labels" : [request_data[i]['labels']]
        }
        df_labels = pd.DataFrame(labels)
        pdList = [df_numerical, df_labels]  # List of dataframes
        final_df = pd.concat(pdList, axis = 'columns')
        model_name = request_data[i]['model']
        #load effort random forest model
        loaded_model = joblib.load(load_path+'/project_control_'+model_name+'_trained.joblib')
        tar_enc = joblib.load(load_path+'/project_control_TargetEnc_trained.joblib')
        values = tar_enc.transform(final_df['labels'])
        final_df = final_df.drop(['labels'], axis = 1)  
        final_df = pd.concat([final_df, values], axis = 1)
        predictions = loaded_model.predict(final_df)
        results.append({
            str(i+1)+' - result' : predictions[0].item()
        })
    return results, 200   

if __name__ == '__main__':
    #DBconnection()
    request_data = [{
               "task_number" : 4,
               "eftm": 0,
               "rdm" : 2,
               "p_type" : 7,
               "department" : 0,
               "business_subfield": 0,
               "priority" : 2,
               "b_funds" : 748512.06,
               "apn": 0,
               "b_duration" : 36,
               "b_cost_type" : 0,
               "wp_number" : 2,
               "wp_type" : 0,
               "wp_duration": 36,
               "labels" : "Literature",
               "model" : "xboost"
               },
               {
               "task_number" : 8,
               "eftm": 0,
               "rdm" : 3,
               "p_type" : 2,
               "department" : 0,
               "business_subfield": 0,
               "priority" : 0,
               "b_funds" : 0,
               "apn": 0,
               "b_duration" : 26,
               "b_cost_type" : 0,
               "wp_number" : 3,
               "wp_type" : 0,
               "wp_duration": 23,
               "labels" : "Others",
               "model" : 'random_forest'
               }
    ]
    # (w.finish - w.start + 1) as wp_duration
    results = predict_from_model(request_data)
    print(results)