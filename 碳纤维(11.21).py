import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
file_path = 'C:/Users/Administrator/Desktop/碳纤维.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
X = df[['掺量', '冲击速度']]
y_compressive = df['动态抗压强度']
y_energy = df['能量']
X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
    X, y_compressive, test_size=0.3, random_state=42
)
X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(
    X, y_energy, test_size=0.3, random_state=42
)
models = {
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    },
    'MLP': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(max_iter=2000, random_state=42, early_stopping=True))
        ]),
        'params': {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__alpha': [0.0001, 0.001, 0.01],
            'mlp__learning_rate_init': [0.001, 0.01]
        }
    },
    'KNN': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ]),
        'params': {
            'knn__n_neighbors': [3, 5, 7, 9],
            'knn__weights': ['uniform'],
            'knn__p': [1, 2]
        }
    },
    'GBDT': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    },
    'AdaBoost': {
        'model': AdaBoostRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 1.0],
            'loss': ['linear', 'square', 'exponential']
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.5]
        }
    },
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {}
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['svd', 'sparse_cg']
        }
    },
    'DecisionTree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
}
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
def kfold_cv_details(estimator, X_data, y_data, cv_splits=10):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    rows = []
    fold_id = 1
    for train_idx, test_idx in kf.split(X_data):
        X_train_cv, X_test_cv = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train_cv, y_test_cv = y_data.iloc[train_idx], y_data.iloc[test_idx]
        model_cv = clone(estimator)
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(X_test_cv)
        fold_metrics = calculate_metrics(y_test_cv, y_pred_cv)
        fold_metrics['Fold'] = fold_id
        rows.append(fold_metrics)
        fold_id += 1
    df_folds = pd.DataFrame(rows)
    df_folds = df_folds[['Fold', 'MSE', 'RMSE', 'MAE', 'R2', 'MAPE']]
    metrics_cols = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']
    mean_values = df_folds[metrics_cols].mean()
    std_values = df_folds[metrics_cols].std(ddof=1)
    mean_row = {'Fold': 'mean'}
    mean_row.update(mean_values.to_dict())
    std_row = {'Fold': 'std'}
    std_row.update(std_values.to_dict())
    df_stats = pd.DataFrame([mean_row, std_row])
    df_all = pd.concat([df_folds, df_stats], ignore_index=True)
    return df_all
results = {
    'compressive': {
        'train_metrics': pd.DataFrame(),
        'test_metrics': pd.DataFrame(),
        'train_preds': pd.DataFrame(),
        'test_preds': pd.DataFrame(),
        'best_params': {},
        'final_model': {},
        'cv_metrics': pd.DataFrame(),
        'learning_curve': {}
    },
    'energy': {
        'train_metrics': pd.DataFrame(),
        'test_metrics': pd.DataFrame(),
        'train_preds': pd.DataFrame(),
        'test_preds': pd.DataFrame(),
        'best_params': {},
        'final_model': {},
        'cv_metrics': pd.DataFrame(),
        'learning_curve': {}
    }
}
sheet_prefix_map = {
    'compressive': '动态抗压强度',
    'energy': '能量'
}
en_prefix_map = {
    'compressive': 'DynamicCompressiveStrength',
    'energy': 'Energy'
}
for target_name, X_train, X_test, y_train, y_test, y_full in [
    ('compressive', X_train_comp, X_test_comp, y_train_comp, y_test_comp, y_compressive),
    ('energy', X_train_energy, X_test_energy, y_train_energy, y_test_energy, y_energy)
]:
    print(f"\nProcessing {target_name}...")
    train_metrics_dict = {}
    test_metrics_dict = {}
    train_preds_dict = {}
    test_preds_dict = {}
    best_params_dict = {}
    final_models = {}
    cv_all_rows = []

    for model_name, model_info in models.items():
        print(f"Training {model_name}...")

        if model_info['params']:
            search = RandomizedSearchCV(
                model_info['model'],
                model_info['params'],
                n_iter=10,
                cv=10,                      # 改为 10 折
                random_state=42,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                return_train_score=True
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params_dict[model_name] = search.best_params_
        else:
            best_model = model_info['model']
            best_model.fit(X_train, y_train)
            best_params_dict[model_name] = {'no_param_tuning': 'default'}
        # 训练集指标
        y_train_pred = best_model.predict(X_train)
        train_preds_dict[model_name] = y_train_pred
        train_metrics_dict[model_name] = calculate_metrics(y_train, y_train_pred)
        # 测试集指标
        y_test_pred = best_model.predict(X_test)
        test_preds_dict[model_name] = y_test_pred
        test_metrics_dict[model_name] = calculate_metrics(y_test, y_test_pred)
        # K 折交叉验证
        cv_df = kfold_cv_details(best_model, X, y_full, cv_splits=10)
        cv_df.insert(0, 'Model', model_name)
        cv_all_rows.append(cv_df)
        final_models[model_name] = best_model
    results[target_name]['train_metrics'] = pd.DataFrame(train_metrics_dict).T
    results[target_name]['test_metrics'] = pd.DataFrame(test_metrics_dict).T
    results[target_name]['train_preds'] = pd.DataFrame(train_preds_dict)
    results[target_name]['test_preds'] = pd.DataFrame(test_preds_dict)
    results[target_name]['best_params'] = best_params_dict
    results[target_name]['final_model'] = final_models
    results[target_name]['cv_metrics'] = pd.concat(cv_all_rows, ignore_index=True)
for target_name, y_full in [
    ('compressive', y_compressive),
    ('energy', y_energy)
]:
    prefix_en = en_prefix_map[target_name]
    for model_name, model_est in results[target_name]['final_model'].items():
        print(f"\nPlotting learning curve: {prefix_en} - {model_name}")
        cv_for_curve = KFold(n_splits=10, shuffle=True, random_state=42)
        train_sizes, train_scores, val_scores = learning_curve(
            model_est,
            X,
            y_full,
            cv=cv_for_curve,
            scoring='r2',
            n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 5),
            shuffle=True,
            random_state=42
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        results[target_name]['learning_curve'][model_name] = {
            'train_sizes': train_sizes,
            'train_scores_mean': train_scores_mean,
            'train_scores_std': train_scores_std,
            'val_scores_mean': val_scores_mean,
            'val_scores_std': val_scores_std
        }
        plt.figure(figsize=(6, 4))
        plt.plot(train_sizes, train_scores_mean, marker='o',
                 label='K-fold CV - training R$^2$')
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.2
        )
        plt.plot(train_sizes, val_scores_mean, marker='o',
                 label='K-fold CV - test R$^2$')
        plt.fill_between(
            train_sizes,
            val_scores_mean - val_scores_std,
            val_scores_mean + val_scores_std,
            alpha=0.2
        )
        plt.xlabel('Number of training samples')
        plt.ylabel('R$^2$')
        plt.title(f'Learning Curve - {prefix_en} ({model_name})')
        plt.legend()
        plt.grid(True)

        curve_path = f'C:/Users/Administrator/Desktop/{prefix_en}_learning_curve_{model_name}.png'
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Learning curve saved to: {curve_path}')
settings_data = [
    ('random_state (train_test_split / KFold / RandomizedSearch)', '42'),
    ('test_size for train_test_split', '0.3'),
    ('KFold splits (cv)', '10'),   # 这里也改成 10
    ('RandomizedSearchCV n_iter', '10'),
    ('scoring', 'neg_mean_squared_error'),
    ('n_jobs', '-1 (use all available CPU cores)')
]
settings_df = pd.DataFrame(settings_data, columns=['Setting', 'Value'])
search_space_rows = []
for model_name, model_info in models.items():
    search_space_rows.append({
        'Model': model_name,
        'Hyperparameter search space': str(model_info['params']) if model_info['params'] else 'None (default parameters)'
    })
search_space_df = pd.DataFrame(search_space_rows)
output_path = 'C:/Users/Administrator/Desktop/模型结果汇总_训练测试集.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for target_name in ['compressive', 'energy']:
        sheet_prefix = sheet_prefix_map[target_name]
        train_metrics = results[target_name]['train_metrics'].copy()
        test_metrics = results[target_name]['test_metrics'].copy()
        best_params = results[target_name]['best_params']
        cv_metrics = results[target_name]['cv_metrics'].copy()
        train_metrics['最佳超参数'] = train_metrics.index.map(lambda x: str(best_params.get(x, {})))
        test_metrics['最佳超参数'] = test_metrics.index.map(lambda x: str(best_params.get(x, {})))
        metric_names = {
            'MSE': '均方误差 (MSE)',
            'RMSE': '均方根误差 (RMSE)',
            'MAE': '平均绝对误差 (MAE)',
            'R2': '决定系数 (R²)',
            'MAPE': '平均绝对百分比误差 (MAPE)'
        }
        train_metrics = train_metrics.rename(columns=metric_names)
        test_metrics = test_metrics.rename(columns=metric_names)
        train_metrics.insert(0, '序号', range(len(train_metrics)))
        test_metrics.insert(0, '序号', range(len(test_metrics)))
        train_metrics.to_excel(writer, sheet_name=f'{sheet_prefix}_训练集评估指标', index=True)
        test_metrics.to_excel(writer, sheet_name=f'{sheet_prefix}_测试集评估指标', index=True)
        train_preds = results[target_name]['train_preds'].copy()
        test_preds = results[target_name]['test_preds'].copy()
        if target_name == 'compressive':
            train_preds.insert(0, '真实值', y_train_comp.values)
            test_preds.insert(0, '真实值', y_test_comp.values)
        elif target_name == 'energy':
            train_preds.insert(0, '真实值', y_train_energy.values)
            test_preds.insert(0, '真实值', y_test_energy.values)
        train_preds.to_excel(writer, sheet_name=f'{sheet_prefix}_训练集预测值', index=False)
        test_preds.to_excel(writer, sheet_name=f'{sheet_prefix}_测试集预测值', index=False)
        cv_metrics = cv_metrics.rename(columns={
            'Model': '模型',
            'Fold': '折次'
        })
        cv_metrics.insert(0, '序号', range(len(cv_metrics)))
        cv_metrics.to_excel(writer, sheet_name=f'{sheet_prefix}_K折交叉验证', index=False)
    settings_df.to_excel(writer, sheet_name='计算设置', index=False)
    search_space_df.to_excel(writer, sheet_name='超参数搜索空间', index=False)
print(f"结果已保存到 {output_path}")
