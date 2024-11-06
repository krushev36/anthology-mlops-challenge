import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder


class NetflixChurnModel:
    model_name = "NetflixModelPipeline"

    def __init__(
        self, mlflow_host=os.getenv("MLFLOW_HOST"), mlflow_port=os.getenv("MLFLOW_PORT")
    ):
        self.pipeline = None
        self.mlflow_host = mlflow_host
        self.mlflow_port = mlflow_port

        mlflow.set_tracking_uri(f"http://{mlflow_host}:{mlflow_port}")

    def load_data(self, data_path=f"{os.getenv('WORKSPACEFOLDER')}/data/netflix.csv"):
        df = pd.read_csv(data_path)
        X = df.drop(
            ["panelist_id", "survival_months", "is_canceled_sum", "censored", "target"],
            axis=1,
        )

        censored = df.censored.tolist()
        survival_months = df.survival_months.tolist()
        y = list(zip(censored, survival_months))
        y = np.array(y, dtype=[("censored", "?"), ("survival_months", "<f8")])

        categorical_columns = X.select_dtypes(include=["object", "bool"]).columns
        X[categorical_columns] = X[categorical_columns].astype("category")
        return X, y

    def preprocess_data(self, X):
        numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["category"]).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(), categorical_features),
            ]
        )
        return preprocessor

    def train(self):
        X, y = self.load_data()
        preprocessor = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=20
        )

        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomSurvivalForest(
                        n_estimators=100,
                        min_samples_split=10,
                        min_samples_leaf=15,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )

        mlflow.set_tracking_uri(f"http://{self.mlflow_host}:{self.mlflow_port}")

        with mlflow.start_run():
            self.pipeline.fit(X_train, y_train)
            c_index = self.pipeline.score(X_test, y_test)
            print(f"Concordance Index (C-index) on test data: {c_index:.4f}")

            mlflow.log_param(
                "model_type", type(self.pipeline.named_steps["model"]).__name__
            )
            mlflow.log_metric("c_index", c_index)
            mlflow.sklearn.log_model(self.pipeline, self.model_name)
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{self.model_name}",
                self.model_name,
            )

        client = mlflow.MlflowClient(f"http://{self.mlflow_host}:{self.mlflow_port}")
        client.set_registered_model_alias(self.model_name, "champion", "1")

    def load(self):
        # Set the MLflow tracking URI
        client = mlflow.MlflowClient(f"http://{self.mlflow_host}:{self.mlflow_port}")

        latest_version_info = client.get_latest_versions(
            self.model_name, stages=["None"]
        )
        latest_version = max([int(version.version) for version in latest_version_info])
        print(f"Predicting with version {latest_version}")

        model_uri = client.get_model_version_download_uri(
            self.model_name, str(latest_version)
        )
        return mlflow.sklearn.load_model(model_uri)

    def predict(self, X_new):
        if self.load() is None:
            raise Exception(
                "Model is not trained yet. Please call the train method first."
            )

        X_new = pd.DataFrame(X_new)
        categorical_columns = X_new.select_dtypes(include=["object", "bool"]).columns
        X_new[categorical_columns] = X_new[categorical_columns].astype("category")

        return self.pipeline.predict(X_new)


# Example usage:
# model = NetflixModel(data_path="/workspaces/antenna-ml-challenge/data/netflix.csv")
# model.train()
# predictions = model.predict(X_new)


if __name__ == "__main__":
    model = NetflixChurnModel()
    model.train()

    sample = {
        "monthly_price_sum": [283.75],
        "monthly_promo_discount_sum": [0.0],
        "zip_code": [11416],
        "distributor": ["Direct"],
        "plan_name": ["Premium"],
        "plan_term_length": ["Monthly"],
        "plan_ads": ["Ad-Free"],
        "subscriber": [True],
        "active_promos": [0],
        "age_group_name": ["45-54 years"],
        "current_service_tenure": ["47"],
        "customer_of_comcast": [False],
        "education_name": ["HighSchool"],
        "ethnicity_name_v2": ["Other"],
        "gender_name": ["M"],
        "has_children": [False],
        "household_income_name_v2": ["$50-100k"],
        "is_gross_add": [False],
    }

    predictions = model.predict(sample)
    print(predictions)
