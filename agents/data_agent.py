import numpy as np
import pandas as pd
from .base_agent import BaseAgent


class DataAgent(BaseAgent):
    """Cleans and preprocesses the uploaded dataset generically."""

    def __init__(self):
        super().__init__("DataAgent")

    def run(self, context):
        raw_df = context["data"].copy()
        df = raw_df.copy()

        # Get target selected by TargetAgent
        target = context["target_column"]

        self.log("Cleaning & preprocessing dataset...")

        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Keep target name consistent
        target = str(target).strip()
        context["target_column"] = target

        # Treat empty strings as NaN
        df.replace({"": np.nan, " ": np.nan}, inplace=True)

        # Drop columns that are almost entirely missing
        # but NEVER drop the target column
        thresh = int(0.9 * len(df))

        cols_to_drop = []

        for col in df.columns:
            if col == target:
                continue

            if df[col].notna().sum() < len(df) - thresh:
                cols_to_drop.append(col)

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # Try to coerce object FEATURE columns to numeric when reasonable
        for col in df.columns:

            # Don't change target datatype here
            if col == target:
                continue

            if df[col].dtype == "object":
                num = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

                if num.notna().sum() >= 0.7 * len(df):
                    df[col] = num

        # Impute missing values
        for col in df.columns:

            # If target is missing, don't invent a target value
            if col == target:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):

                mean_val = df[col].mean()

                df[col] = df[col].fillna(
                    mean_val
                )

            else:

                if df[col].isna().any():

                    try:
                        mode = df[col].mode().iloc[0]

                    except IndexError:
                        mode = "Unknown"

                    df[col] = df[col].fillna(
                        mode
                    )

        # Remove rows where target is missing
        if target in df.columns:
            df = df[df[target].notna()].copy()

        # One-hot encode categorical FEATURES only
        cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # VERY IMPORTANT:
        # do not encode the target
        cat_cols = [
            col for col in cat_cols
            if col != target
        ]

        if cat_cols:
            df = pd.get_dummies(
                df,
                columns=cat_cols,
                drop_first=True
            )

        context["raw_data"] = raw_df
        context["clean_data"] = df

        self.log("Data preprocessing complete")

        return context
