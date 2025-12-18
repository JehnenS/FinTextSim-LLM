import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

class PrepML:
    def __init__(self, df, target_name: str):
        """
        Initialize the function
        Format all infinite values and ensure integer format for target for safety
        """
        self.target_name = target_name
        self.df = df

        #for safety: format all values which contrain infinite values into nan
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df[self.target_name] = self.df[self.target_name].astype(int) #ensure integer format for target

    def split_target_features(self):
        """
        Split full dataframe into  X (predictors) and y (target) - do it on CPU to let xgboost handle missing values
        """
    
        print(f"Number of NA values in full df: {self.df.isna().sum().sum()}")
        print(f"Share of NA values in full df: {(self.df.isna().sum().sum()) / self.df.size * 100:.2f}%")
        
        
        # Define features and target
        X = self.df.drop(columns=[self.target_name])
        print(f"Number of NA values in columns: {X.isna().sum().sum()}")
        print(f"Share of NA values in columns: {(X.isna().sum().sum()) / X.size * 100:.2f}%")
        print(f"Number of columns: {X.shape[1]}")
        
        #X.fillna(X.mean(numeric_only = True), inplace = True)
        y = self.df[self.target_name]

        neg_instances = (y == 0).sum()
        pos_instances = (y == 1).sum()
    
        return X, y, neg_instances, pos_instances



    def replace_placeholder_values(
        self,
        X: pd.DataFrame,
        categorical_cols: list,
        placeholder_values: list = [-1, -999, 999, 9999, -9999]
    ):
        """
        Replace placeholder values (e.g., -1, 9999) with NaN for all numeric columns.
        Also prints how many replacements were made in total and per column.
        """
        
        X_cleaned = X.copy()
    
        # Select numeric columns (excluding categorical)
        numeric_cols = [
            col for col in X_cleaned.columns
            if col not in categorical_cols and pd.api.types.is_numeric_dtype(X_cleaned[col])
        ]
    
        total_replaced = 0
        col_replacements = {}
    
        for col in numeric_cols:
            mask = X_cleaned[col].isin(placeholder_values)
            n_replaced = mask.sum()
            if n_replaced > 0:
                X_cleaned.loc[mask, col] = np.nan
                col_replacements[col] = n_replaced
                total_replaced += n_replaced
    
        # ----- Reporting -----
        if total_replaced == 0:
            print(f"No placeholder values {placeholder_values} found.")
        else:
            print(f"Replaced {total_replaced:,} placeholder values with NaN "
                  f"across {len(col_replacements)} columns.")
            # Optional: print top 5 most affected columns
            top_cols = dict(sorted(col_replacements.items(), key=lambda x: x[1], reverse=True)[:5])
            print("Top affected columns (most replacements):", top_cols)
    
        return X_cleaned

    def replace_outliers_with_nan(self, X, categorical_cols:list):
        """
        Replace outliers in X numeric columns with NaN, using the IQR boxplot rule.
        Returns a new DataFrame.
        """
        X_cleaned = X.copy()
        
        numeric_cols = [col for col in X_cleaned.columns if col not in categorical_cols and pd.api.types.is_numeric_dtype(X_cleaned[col])]
    
        for col in numeric_cols:
            series = X_cleaned[col].dropna()
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_whisker = Q1 - 1.5 * IQR
            upper_whisker = Q3 + 1.5 * IQR
    
            outlier_mask = (X_cleaned[col] < lower_whisker) | (X_cleaned[col] > upper_whisker)
            X_cleaned.loc[outlier_mask, col] = np.nan
    
        return X_cleaned



    def scale_features(self, X_train, X_test, categorical_cols:list):
        """
        function scale features in a df
        Avoid scaling categorical features and the target variable
        Use the scaler from the train-set on test-set to avoid data leakage
        """
        from sklearn.preprocessing import StandardScaler
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        scaler = StandardScaler()
        feature_cols = [col for col in X_train.columns if col not in categorical_cols]
        X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
        X_test[feature_cols] = scaler.transform(X_test[feature_cols])
        print("Features scaled")
        return X_train, X_test

    def remove_irrelevant_columns(self, X, cols_to_exclude:list):
        """
        Remove columns which are not numerical, such as dates, etc.
        Remove columns from cols_to_exclude
        """
        X = X.copy()
        X = X.select_dtypes(include='number')
        X = X.drop(columns = cols_to_exclude, errors = "ignore") #drop year column
        
        return X

    def _year_based_split(self, X, y, test_start_year):
        """
        Split train/test by years to avoid data leakage.
        """
        if "year" not in X.columns: #self.df.columns
            raise KeyError("Year column not found in dataframe. Ensure your df has a 'year' column.")
    
        years = X["year"]  # get year from original df - self.df["year"]
        
        # Split based on threshold year
        train_mask = years < test_start_year
        test_mask = years >= test_start_year
    
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    
        print(f"Training years: {sorted(X_train['year'].unique())}")
        print(f"Testing years:  {sorted(X_test['year'].unique())}")
    
        return X_train, X_test, y_train, y_test


    def drop_rows_with_problematic_values(self, X, y, problematic_values=[np.nan, 0, -1], threshold=0.6, ): 
        """ 
        Drop rows with too many problematic values (e.g., NaN, 0, -1). 
        
        Parameters: 
        ----------- 
        X, y : pd.DataFrame Input dataframe. 
        problematic_values : list, default [np.nan, 0, -1] List of values considered problematic. 
        threshold : float, default 0.6 Maximum allowed share of problematic values in a row before dropping. For example, 0.6 means drop rows where 60% or more of the columns contain problematic values. 
        
        Returns: 
        -------- 
        pd.DataFrame : filtered dataframe with problematic rows removed. 
        """ 
        X = X.copy() 

        #filter non-sentiment columns
        non_sentiment_columns = [col for col in X.columns if "_sentiment" not in col]
        
        # Start with NaN share 
        problematic_share = X[non_sentiment_columns].isna().astype(float) 
        
        # Add shares for each value 
        for val in problematic_values: 
            if pd.isna(val):
                continue # already counted NaNs 
            mask = (X == val).astype(float) 
            problematic_share += mask 
        
        # Convert to proportion of problematic values per row 
        problematic_share = problematic_share.mean(axis=1) 
        
        # Filter rows below threshold 
        mask = problematic_share < threshold 
        dropped = (~mask).sum() 
        print(f"Dropped {dropped} rows due to >={threshold*100:.1f}% problematic values ({problematic_values}).") 
        X_clean = X.loc[mask]#.reset_index(drop=True) 
        y_clean = y.loc[mask]#.reset_index(drop=True) 
        print(f"Shape of X_clean: {X_clean.shape}")
        print(f"Shape of y_clean: {y_clean.shape}")
        
        return X_clean, y_clean

    
    
    def drop_columns_with_problematic_values(
        self,
        X,
        problematic_values=[np.nan, 0, -1],
        threshold=0.5,
    ):
        """
        Drop columns that contain too many problematic values (e.g., NaN, 0, -1).
    
        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.
        problematic_values : list, default [np.nan, 0, -1]
            Values considered problematic.
        threshold : float, default 0.5
            Drop columns where the share of problematic values >= threshold.
            For example, 0.5 means drop columns with 50% or more problematic values.
    
        Returns
        -------
        pd.DataFrame : Cleaned dataframe.
        """
    
        X = X.copy()
    
        # Start with NaN mask
        problematic_mask = X.isna().astype(float)
    
        # Add masks for specific values
        for val in problematic_values:
            if pd.isna(val):
                continue
            problematic_mask += (X == val).astype(float)
    
        # Compute share of problematic values per column
        problematic_share = problematic_mask.mean(axis=0)
    
        # Identify columns to drop
        cols_to_drop = problematic_share[problematic_share >= threshold].index.tolist()
    
        print(f"\nDropped {len(cols_to_drop)} columns with >= {threshold*100:.1f}% problematic values ({problematic_values}).")
    
        X = X.drop(columns=cols_to_drop)
        
        print(f"Remaining columns: {X.shape[1]}")
        print(f"Remaining NA share: {X.isna().sum().sum() / X.size * 100:.2f}%")
    
        return X

    

    def _oversample_training_data(self, X_train, y_train, method: str = "smote", random_state: int = 42):
        """
        Optionally oversample the training set to balance classes.
        Supports 'smote' and 'random' methods.
        """
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        
        print(f"Applying oversampling: {method}")
        if method == "smote":
            sampler = SMOTE(random_state=random_state)
        elif method == "random":
            sampler = RandomOverSampler(random_state=random_state)
        else:
            raise ValueError(f"Unsupported oversampling method: {method}. Use 'smote' or 'random'.")
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"Before oversampling: {np.bincount(y_train)}")
        print(f"After oversampling:  {np.bincount(y_resampled)}")
        print(f"Resampled training shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled


    def report_data_characteristics(self, X, y, name = "train"):
        """
        Report characteristics of the train data
        """
        print(f"\nData characteristics - {name}")
        print(f"Shape X: {X.shape}")
        print(f"Share of missing values: {X.isna().mean().mean():.2%}")
        print(f"Share of zeros: {(X == 0).mean().mean():.2%}")
        print(f"# Positives: {(y == 1).sum()}; # Negatives: {(y == 0).sum()}")
        print(f"Class balance: {(y == 1).mean():.2%} positive, {(y == 0).mean():.2%} negative\n")

    def run_lr_preprocessing(
        self, 
        categorical_cols:list, 
        cols_to_exclude: list, 
        test_start_year:int = 2022,
        placeholder_values:list = [-1, 999, -999, 9999, -9999],
        problematic_values:list = [np.nan, -1, 0],
        threshold_columns = 0.6,
        threshold_rows = 0.6,
        fill_na: float | None = 0,
        oversample_method: str | None = None  # smote or random oversampling of training data - or no oversampling
    ):
        """
        Wrapper function to run preprocessing for Logistic Regression (cuml)

        Very high thresholds for na and zero effectively disable column dropping
        """
        #split target features
        X, y, neg_instances, pos_instances = self.split_target_features() #split df into features and target dfs
        X = self.remove_irrelevant_columns(X, cols_to_exclude) #remove irrelevant columns

        #0. remove rows which have too many problematic values --> before splitting into test/train to ensure good learning
        X, y = self.drop_rows_with_problematic_values(X, y, problematic_values, threshold = threshold_rows)

        #1. perform train-test-split before every other preprocessing step to avoid data leakage, then drop year column
        X_train, X_test, y_train, y_test = self._year_based_split(X, y, test_start_year)
        X_train = X_train.drop(columns=["year"], errors="ignore")
        X_test = X_test.drop(columns=["year"], errors="ignore")

        #2. replace placeholder values on train- and testset separately --> avoid data leakage
        X_train = self.replace_placeholder_values(X_train, categorical_cols = categorical_cols, placeholder_values = placeholder_values)
        X_test = self.replace_placeholder_values(X_test, categorical_cols = categorical_cols, placeholder_values = placeholder_values)

        #3. replace outliers on train- and testset separately --> avoid data leakage
        X_train = self.replace_outliers_with_nan(X_train, categorical_cols)
        X_test = self.replace_outliers_with_nan(X_test, categorical_cols)

        #4. drop columns with problematic values
        X_train = self.drop_columns_with_problematic_values(X_train, problematic_values, threshold = threshold_columns)
        X_test = X_test[X_train.columns] #keep same columns

        #5. Fill remaining NaNs
        if fill_na is not None:
            X_train = X_train.fillna(fill_na)
            X_test = X_test.fillna(fill_na)
            print(f"Filled remaining NaN values with {fill_na}.")
        else:
            print("Skipped NA filling (may leave missing values for model to handle).")

        # --- Optional oversampling ---
        if oversample_method is not None:
            X_train, y_train = self._oversample_training_data(X_train, y_train, method=oversample_method)
        else:
            print("No oversampling applied.")
        
        #6. scale features
        X_train, X_test = self.scale_features(X_train, X_test, categorical_cols)

        print(f"\nNumber of remaining predictors: {X_train.shape[1]}")
        print(f"\nNumber of train-instances: {X_train.shape[0]}")
        print(f"Number of test-instances: {X_test.shape[0]}")
        print(f"Train-share: {X_train.shape[0] / X.shape[0] * 100:.2f}%\n")

        self.report_data_characteristics(X_train, y_train)
        self.report_data_characteristics(X_test, y_test, name = "test")
        self.report_data_characteristics(X, y, name = "Full")

        return X_train, X_test, y_train, y_test, neg_instances, pos_instances, X, y

    def run_tree_preprocessing(
        self, 
        categorical_cols:list, 
        cols_to_exclude: list, 
        test_start_year:int = 2022,
        placeholder_values:list = [-1, 999, -999, 9999, -9999],
        problematic_values:list = [np.nan, -1, 0],
        threshold_columns = 0.6,
        threshold_rows = 0.6,
        scale_features:bool = False,
        oversample_method: str | None = None  # smote or random oversampling of training data - or no oversampling
    ):
        """
        Wrapper function to run preprocessing for tree-based models (xgb package for both RF and XGB)

        Very high thresholds for na and zero effectively disable column dropping
        Dissimilarity to LR processing: 
            - No imputation of missing values as XGB handles these itself
            - No scaling needed as models are tree-based
        """
        #split target features
        X, y, neg_instances, pos_instances = self.split_target_features() #split df into features and target dfs
        X = self.remove_irrelevant_columns(X, cols_to_exclude) #remove irrelevant columns

        #0. remove rows which have too many problematic values --> before splitting into test/train to ensure good learning
        X, y = self.drop_rows_with_problematic_values(X, y, problematic_values, threshold = threshold_rows)

        #1. perform train-test-split before every other preprocessing step to avoid data leakage, then drop year column
        X_train, X_test, y_train, y_test = self._year_based_split(X, y, test_start_year)
        X_train = X_train.drop(columns=["year"], errors="ignore")
        X_test = X_test.drop(columns=["year"], errors="ignore")

        #2. replace placeholder values on train- and testset separately --> avoid data leakage
        X_train = self.replace_placeholder_values(X_train, categorical_cols = categorical_cols, placeholder_values = placeholder_values)
        X_test = self.replace_placeholder_values(X_test, categorical_cols = categorical_cols, placeholder_values = placeholder_values)

        #3. replace outliers on train- and testset separately --> avoid data leakage
        X_train = self.replace_outliers_with_nan(X_train, categorical_cols)
        X_test = self.replace_outliers_with_nan(X_test, categorical_cols)

        #4. drop columns with problematic values
        X_train = self.drop_columns_with_problematic_values(X_train, problematic_values, threshold = threshold_columns)
        X_test = X_test[X_train.columns] #keep same columns

        # --- Optional oversampling ---
        if oversample_method is not None:
            print(f"Applying oversampling with temporary NA filling (method={oversample_method})")
            # store NaN positions
            nan_mask = X_train.isna()
        
            # fill temporarily for oversampling
            X_train_filled = X_train.fillna(0)
        
            # perform oversampling
            X_train_res, y_train_res = self._oversample_training_data(X_train_filled, y_train, method=oversample_method)
        
            # restore NaNs at the original positions (only valid for non-synthetic rows)
            # for RandomOverSampler: easy → because it replicates existing samples
            if oversample_method == "random":
                mask_resampled = nan_mask.reindex(X_train_res.index, fill_value=False)
                X_train_res.loc[:, :] = np.where(mask_resampled, np.nan, X_train_res)
        
            # for SMOTE: cannot directly restore NaNs (synthetic samples are interpolated)
            elif oversample_method == "smote":
                # we can only restore NaNs in columns where all contributing samples had NaNs
                # otherwise the interpolated values are valid numeric data
                print("Note: SMOTE interpolation means missing values cannot be fully restored — only where NaNs dominate.")
                for col in X_train.columns:
                    if nan_mask[col].all():
                        X_train_res[col] = np.nan
        
            X_train = X_train_res
            y_train = y_train_res

            print(f"Oversampling complete. Training size: {X_train.shape}")
        else:
            print("No oversampling applied.")

        print(f"\nNumber of remaining predictors: {X_train.shape[1]}")
        print(f"\nNumber of train-instances: {X_train.shape[0]}")
        print(f"Number of test-instances: {X_test.shape[0]}")
        print(f"Train-share: {X_train.shape[0] / X.shape[0] * 100:.2f}%\n")

        #6. scale features
        if scale_features:
            X_train, X_test = self.scale_features(X_train, X_test, categorical_cols)


        self.report_data_characteristics(X_train, y_train)
        self.report_data_characteristics(X_test, y_test, name = "test")
        self.report_data_characteristics(X, y, name = "Full")

        
        return X_train, X_test, y_train, y_test, neg_instances, pos_instances, X, y
