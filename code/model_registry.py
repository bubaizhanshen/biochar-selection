"""Fixed regression families; optional boosting packages are imported on demand."""
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

DEFAULT_MODELS = ('ridge', 'svr', 'random_forest')
SUPPORTED_MODELS = (*DEFAULT_MODELS, 'extra_trees', 'gradient_boosting',
                    'xgboost', 'lightgbm', 'catboost', 'knn')


def configured_models(config):
    names = tuple(config.get('models', DEFAULT_MODELS))
    if not names or len(names) != len(set(names)):
        raise ValueError('Model order must be nonempty and contain no duplicates')
    unknown = set(names).difference(SUPPORTED_MODELS)
    if unknown:
        raise ValueError(f'Unknown regression families: {sorted(unknown)}')
    if any(name not in config for name in names):
        raise ValueError('Every model requires an explicit parameter dictionary')
    return names


def make_regressor(name, parameters):
    classes = {'ridge': Ridge, 'svr': SVR, 'random_forest': RandomForestRegressor,
               'extra_trees': ExtraTreesRegressor, 'gradient_boosting': GradientBoostingRegressor,
               'knn': KNeighborsRegressor}
    if name == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(**parameters)
    if name == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**parameters)
    if name == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**parameters)
    if name not in classes:
        raise ValueError(f'Unknown regression family: {name}')
    return classes[name](**parameters)
