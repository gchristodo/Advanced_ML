from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from collections import defaultdict
import pandas as pd
import numpy as np


def save_to_excel(file_name, r2, mse, mae, model_names = [], params = []):
    model_name_params = defaultdict(list)
    all_params = defaultdict(list)
    
    for model_name in model_names:
        model_name_params['Estimator'].append(model_name)
        
    for param_dict in params:
        for param_name, param_val in param_dict.items():
            param_name = param_name.split('__')[-1]
            all_params[param_name].append(param_val)
            
    df_model = pd.DataFrame(list(zip(*model_name_params.values(), *all_params.values(),
                                     r2, mse, mae)),
                 columns=(*model_name_params.keys(), *all_params.keys(),
                          'R2', 'MSE', 'MAE'))
    df_model.to_excel(file_name + '.xlsx')


def print_scores(r2, mse, mae):
    print('R2:', r2)
    print('MSE:', mse)
    print('MAE:', mae)


df = pd.read_csv('fuel_emissions.csv', dtype={'transmission_type': str})

#df.info()
pd.DataFrame(df.dtypes).to_excel('dtypes.xlsx')
df.count().to_excel('count.xlsx')

y_column = 'fuel_cost_12000_miles'
columns_to_drop = [y_column, 'file', 'model', 'description']
categorical_columns = ['manufacturer', 'transmission', 'transmission_type', 'fuel_type']
ordinal_columns = ['tax_band']

# label is missing, delete whole line
df.dropna(axis=0, subset=[y_column], inplace=True)

#df.info()

X = df.drop(columns_to_drop, axis=1)
y = df[y_column]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipeline_cat = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

pipeline_ord = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder()),
    ('scaler', MinMaxScaler()),
    ])

pipeline_num = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ])

column_trans = ColumnTransformer([
    ('cat', pipeline_cat, categorical_columns),
    ('ord', pipeline_ord, ordinal_columns),
    ], 
    remainder=pipeline_num)

dtr = DecisionTreeRegressor(random_state=0)

pipeline_pre = Pipeline([
    ('transformer', column_trans),
    ('select', SelectFromModel(dtr, threshold=1e-3)),
    ])

scoring = {
    'R2': 'r2',
    'MSE': 'neg_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
}

models = [
    ('Linear Regression', LinearRegression(), {}),
    ('Decision Tree Regressor', dtr, {
        'model__max_depth': np.arange(3, 23, 2),
        }),
    ('Random Forest Regressor', RandomForestRegressor(random_state=0), {
        'model__n_estimators': (10, 20, 50, 100),
        'model__max_depth': np.arange(3, 23, 2),
        }),
    ('Bagging Regressor', BaggingRegressor(base_estimator=dtr, random_state=0), {
        'model__n_estimators': (10, 20, 50, 100),
        'model__base_estimator__max_depth': np.arange(3, 23, 2),
        }),
    ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=0), {
        'model__n_estimators': (100, 200, 500, 1000, 1500),
        'model__loss': ('ls', 'lad', 'huber'),
        }),
    ]

# just to see the dimensions
x_trans_1 = column_trans.fit_transform(x_train, y_train)
x_trans_2 = pipeline_pre.fit_transform(x_train, y_train)
print('Dimensions: {} -> {} -> {}'.format(x_train.shape[1], x_trans_1.shape[1],
                                     x_trans_2.shape[1]))
print()

best_r2_scores = []
best_mse_scores = []
best_mae_scores = []

for model_name, model, param_grid in models:
    print('Estimator:', model_name)
    
    pipeline = Pipeline([
        ('pre', pipeline_pre),
        ('model', model),
    ])
    
    gs = GridSearchCV(pipeline, param_grid, refit='R2',
                      scoring=scoring, n_jobs=-1, verbose=2)
    gs.fit(x_train, y_train)
    
    results = gs.cv_results_
    save_to_excel(model_name, results['mean_test_R2'], -results['mean_test_MSE'],
                  -results['mean_test_MAE'], params=results['params'])
    
    y_pred = gs.predict(x_test)
    
    best_r2_scores.append(r2_score(y_test, y_pred))
    best_mse_scores.append(mean_squared_error(y_test, y_pred))
    best_mae_scores.append(mean_absolute_error(y_test, y_pred))
    
    print('Best parameters:', gs.best_params_)
    print_scores(best_r2_scores[-1], best_mse_scores[-1], best_mae_scores[-1])
    print()

model_names = [x[0] for x in models]
save_to_excel('best_models', best_r2_scores, best_mse_scores, best_mae_scores,
              model_names=model_names)

