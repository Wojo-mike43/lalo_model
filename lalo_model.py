import datetime as dt
import pandas as pd
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

fred_ids = [
    "FEDFUNDS", "DGS10", "DGS2", "CPIAUCSL", "UNRATE",
    "PAYEMS", "ICSA", "INDPRO", "HOUST", "PCE",
    "BAA10Y", "UMCSENT", "BUSINV", "PI", "PSAVERT",
    "M2SL", "SP500", "FGEXPND", "BOPGSTB", "DRSFRMACBS",
    "EXHOSLUSM495S", "CSUSHPINSA", "DCOILWTICO", "CFNAIMA3", "TCU",
    "IR", "IEABC", "VIXCLS", "TOTALSL", "MTSDS133FMS",
    "EFFR", "NCBDBIQ027S", "DSPIC96", "REVOLSL", "DRCCLACBN",
    "CORCCACBN", "DRCLACBN", "CFNAIDIFF", "CIVPART", "WALCL",
    "GFDEGDQ188S", "A053RC1Q027SBEA", "OPHNFB", "JTSJOL", "GFDEBTN",
    "CORESTICKM159SFRBATL", "GDPC1", "DRCCLOBS", "DRCCLT100S"
]


class DataFetcher:
    def __init__(self, api_key, data_ids):
        if api_key == "your FRED API Key here":
            raise ValueError('Please provide your FRED API key')
        self.fred = Fred(api_key= api_key)
        self.data_ids = data_ids

    def fetch_data_fred_multi(self, days, frequency='D', end_date=None):
        end_date = pd.to_datetime(end_date) if end_date else dt.datetime.today()
        start_date = end_date - dt.timedelta(days)

        df_combined = pd.DataFrame()

        for data_id in self.data_ids:
            df = self.fred.get_series(data_id, start_date=start_date, end_date=end_date)
            if df is not None:
                df = df[df.index.to_series().between(start_date, end_date)]
                df.index = pd.to_datetime(df.index)
                df_resampled = df.resample(frequency).ffill()
                df_combined[data_id] = df_resampled

        return df_combined


class DataClean:
    def __init__(self, data):
        self.data = data

    def data_clean(self):
        self.data.interpolate(method='linear', limit_direction='forward', inplace=True)
        data = self.data.dropna(axis=1)
        data['rate_cuts'] = (data['FEDFUNDS'].shift(1) - data['FEDFUNDS'] > .24).astype(int)
        feature_data = data.drop(['rate_cuts', 'FEDFUNDS', 'CFNAIMA3', 'CFNAIDIFF'], axis=1)

        titles = feature_data.columns
        new_columns = []

        for title in titles:
            pct_col_name = f"{title}_pctchange"
            shift_1_col_name = f"{title}_shift1"
            shift_3_col_name = f"{title}_shift3"
            shift_6_col_name = f"{title}_shift6"

            new_columns.append(feature_data[title].pct_change().rename(pct_col_name))
            new_columns.append(feature_data[title].shift(1).rename(shift_1_col_name))
            new_columns.append(feature_data[title].shift(3).rename(shift_3_col_name))
            new_columns.append(feature_data[title].shift(6).rename(shift_6_col_name))

        changed_data = round(pd.concat([feature_data] + new_columns, axis=1), 4)
        changed_data = changed_data.dropna(axis=0)
        model_data = pd.concat([changed_data, data['rate_cuts']], axis=1).dropna()

        return model_data


class FeatureSelector:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def recursive_feature_elim(self):
        rf = RandomForestClassifier(random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rfecv = RFECV(estimator=rf, step=1, cv=skf, scoring='recall')
        rfecv.fit(self.X_train, self.y_train)

        selected_features = self.X_train.columns[rfecv.support_]
        feature_importance = rfecv.estimator_.feature_importances_

        selected_features_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

        return selected_features_df


class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_model(self):
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [3, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'bootstrap': [True, False],
            'class_weight': [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}]
        }

        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1, verbose=2, scoring='recall')
        grid_search.fit(self.X_train, self.y_train)
        best_rf = grid_search.best_estimator_
        print(f"Best Hyperparameters: {grid_search.best_params_}")

        y_pred = best_rf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))


if __name__ == '__main__':

    #Please input your FRED API key below:
    datafetch = DataFetcher(api_key='your FRED API Key here', data_ids=fred_ids)
    data = datafetch.fetch_data_fred_multi(days=17000)
    print('Data Pull Complete')

    datacleaner = DataClean(data)
    cleaned_data = datacleaner.data_clean()
    print("Data Cleaning Complete")

    X = cleaned_data.drop(['rate_cuts'], axis=1)
    y = cleaned_data['rate_cuts']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=True)
    print("Train-Test Split Complete")

    featureselector = FeatureSelector(X_train, y_train)
    selected_features = featureselector.recursive_feature_elim()
    print('Recursive Feature Elimination Complete')

    X_train = X_train[selected_features['Feature']]
    X_test = X_test[selected_features['Feature']]

    model_trainer = ModelTrainer(X_train, y_train, X_test, y_test)
    best_model = model_trainer.train_model()

