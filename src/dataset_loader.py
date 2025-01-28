import os
from typing import Any, Optional

import pandas as pd
import streamlit as st
from pandas import DataFrame

from config import DEFAULT_SETTINGS


class DatasetLoader:
    """
    A class to manage dataset loading and preprocessing for predefined and custom datasets.
    """

    def __init__(self) -> None:
        os.makedirs(DEFAULT_SETTINGS["predefined_dataset_path"], exist_ok=True)
        os.makedirs(DEFAULT_SETTINGS["custom_dataset_path"], exist_ok=True)

    def load_dataset(
        self,
        uploaded_file: Optional[Any] = None,
        predefined_name: Optional[str] = None,
    ) -> Optional[DataFrame]:
        """
        Loads a dataset from an uploaded file or a predefined dataset name.

        Args:
            uploaded_file (Optional[Any], optional): The uploaded file object (CSV or JSON).
            predefined_name (Optional[str], optional): The name of the predefined dataset to load.

        Returns:
            Optional[DataFrame]: A Pandas DataFrame containing the loaded dataset,
            or None if loading fails or no valid input is provided.
        """
        if uploaded_file:
            return self._handle_uploaded_file(uploaded_file)
        elif predefined_name:
            return self._load_predefined(predefined_name)
        return None

    def _handle_uploaded_file(self, file: Any) -> Optional[DataFrame]:
        """
        Handles the processing of an uploaded dataset file.

        Args:
            file (Any): The uploaded file object to be processed.

        Returns:
            Optional[DataFrame]: A Pandas DataFrame containing the dataset if successfully loaded,
            or None if the file format is unsupported or validation fails.
        """
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".json"):
                df = pd.read_json(file)
            else:
                st.error("Unsupported file format. Use CSV or JSON.")
                return None

            required = {"question", "ground_truth"}
            if not required.issubset(df.columns):
                st.error(f"Dataset must contain: {', '.join(required)} columns")
                return None

            parquet_path = f"data/custom/{file.name.split('.')[0]}.parquet"
            df.to_parquet(parquet_path)
            return df

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

    def _load_predefined(self, name: str) -> Optional[DataFrame]:
        """
        Loads a predefined dataset by its name.

        Args:
            name (str): The name of the predefined dataset to load.

        Returns:
            Optional[DataFrame]: A Pandas DataFrame containing the loaded dataset,
            or None if the dataset cannot be found or loaded.
        """
        try:
            path = f"data/predefined/{name}.parquet"
            return pd.read_parquet(path)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None
