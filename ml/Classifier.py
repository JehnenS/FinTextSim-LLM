import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from cuml.linear_model import LogisticRegression
import cupy as cp
import shap


class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test, neg_instances, pos_instances, X, y):
        """
        Initialize values
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.neg_instances = neg_instances
        self.pos_instances = pos_instances
        self.X = X
        self.y = y


    def compute_shap_values_gpu(self, model, max_display=10):
        """
        Compute SHAP values using GPU acceleration for tree-based models.
        """
        print("Computing SHAP values using GPU acceleration")

        # Initialize GPU TreeExplainer
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", gpu=True)

        # Compute SHAP values for test set
        shap_values = explainer.shap_values(self.X_test)

        # Compute mean absolute importance per feature
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": self.X_test.columns,
            "mean_abs_shap": shap_importance
        }).sort_values("mean_abs_shap", ascending=False)

        # Print summary
        print("\nTop SHAP features:")
        print(shap_df.head(max_display))

        return explainer, shap_values, shap_df

    def lr_classification(self, max_iter: int = 1000, C: float = 1.0, penalty:str = "l2", tol:float = 1e-4, class_weight:str = "balanced"):
        """
        Fit Linear Regression model (cuml) and predict outcomes as well as probabilities
        """
        from cuml.linear_model import LogisticRegression
        
        model = LogisticRegression(max_iter=max_iter, #max number of iterations taken for solver to converge
                                   C=C, #inverse of regularization strength
                                   penalty=penalty, 
                                   tol=tol, #tolerance for stopping criteria
                                   class_weight = class_weight, #ensure balanced learning --> based on occurrences of class labels
                                   )
        
        model.fit(self.X_train, self.y_train)
        print("Logistic Regression model fitted.")
        
        #----------Foundation for evaluation
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Convert to probabilities for log loss
        y_train_prob = model.predict_proba(self.X_train)
        y_train_prob = y_train_prob.iloc[:, 1] #extract only probability for class 1
        y_test_prob = model.predict_proba(self.X_test)
        y_test_prob = y_test_prob.iloc[:, 1]

        return y_train_pred, y_test_pred, y_train_prob, y_test_prob

    def rf_classification(
        self, 
        tree_method:str = "hist",
        objective:str = "binary:logistic",
        eval_metric: str = "aucpr",
        subsample:float = 0.6,
        colsample_bytree:float = 1.0,
        max_depth:int = 6,
        num_parallel_tree: int = 500,
        reg_lambda: float = 1.0,
        reg_alpha:float = 0.2,
        classification_threshold:float = 0.5, #probability needed to classify as positive
    ):
        """
        Perform classification with RF model based on XGB package
        """
        import xgboost as xgb
        
        # Prepare DMatrix (supports GPU acceleration internally)
        dtrain = xgb.DMatrix(self.X_train, label = self.y_train, missing = np.nan)
        dtest = xgb.DMatrix(self.X_test, label = self.y_test, missing = np.nan)

        neg_instances = (self.y_train == 0).sum()
        pos_instances = (self.y_train == 1).sum()
        scale_pos_weight = (neg_instances / pos_instances) * 1
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"Number of instances in train set: {len(self.y_test)}")

         #--------------fit the model
        params = {
            "booster": "gbtree", #as we are training trees
            "tree_method": tree_method,
            "device": "cuda",
            "objective": objective,
            "eval_metric": eval_metric,  # or 'aucpr' etc.
            #"min_child_weight": min_child_weight,
            "learning_rate": 1.0,               # Needed to emulate RF (no boosting) (only for regression?!)  - https://xgboost.readthedocs.io/en/stable/tutorials/rf.html#random-forests-tm-in-xgboost
            "subsample": subsample,
            "colsample_bytree": colsample_bytree, #include all columns to enforce feature interaction
            "max_depth": max_depth,
            "num_parallel_tree": num_parallel_tree,        
            "lambda": reg_lambda,
            "alpha": reg_alpha,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42
        }

        # Train
        model = xgb.train(
            params = params,
            dtrain = dtrain,
            num_boost_round = 1, #ensure that not multiple forests are trained --> no boosting and RF emulation
        )
            
        print("RF model fitted.")
        y_test_prob = model.predict(dtest)
        y_test_pred = (y_test_prob >= classification_threshold).astype(int)

        y_train_prob = model.predict(dtrain)
        y_train_pred = (y_train_prob >= classification_threshold).astype(int)
        print("Prediction finished\n")

        return y_train_pred, y_test_pred, y_train_prob, y_test_prob

    
    def xgb_classification(
        self, 
        tree_method:str = "hist",
        objective:str = "binary:logistic",
        eval_metric: str = "aucpr",
        num_boost_round = 500,
        min_child_weight:int = 15,
        learning_rate:float = 0.05,
        subsample:float = 0.6,
        colsample_bytree:float = 1.0,
        max_depth:int = 6,
        num_parallel_tree: int = 1, #Needs to be 1 as a value > 1 may trigger RF behavior
        reg_lambda: float = 1.0,
        reg_alpha:float = 0.2,
        classification_threshold:float = 0.5, #probability needed to classify as positive
        perform_cv:bool = True, #perform cross-validation to determine best_n_estimators or not
        scale_pos_weight_factor:float = 1.0 #factor for scaling weights on classes
    ):
        """
        Perform classification with SGB model based on XGB package
        """
        import xgboost as xgb
        
        # Prepare DMatrix (supports GPU acceleration internally)
        dtrain = xgb.DMatrix(self.X_train, label = self.y_train, missing = np.nan)
        dtest = xgb.DMatrix(self.X_test, label = self.y_test, missing = np.nan)

        neg_instances = (self.y_train == 0).sum()
        pos_instances = (self.y_train == 1).sum()
        scale_pos_weight = (neg_instances / pos_instances) * scale_pos_weight_factor
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"Number of instances in train set: {len(self.y_test)}")

         #--------------fit the model
        params = {
            "booster": "gbtree", #as we are training trees
            "tree_method": tree_method,
            "device": "cuda",
            "objective": objective,
            "eval_metric": eval_metric,  # or 'aucpr' etc.
            "min_child_weight": min_child_weight,
            "learning_rate": learning_rate,               # Needed to emulate RF (no boosting) (only regression?!)  - https://xgboost.readthedocs.io/en/stable/tutorials/rf.html#random-forests-tm-in-xgboost
            "subsample": subsample,
            "colsample_bytree": colsample_bytree, #include all columns to enforce feature interaction
            "max_depth": max_depth,
            "num_parallel_tree": num_parallel_tree,        
            "lambda": reg_lambda,
            "alpha": reg_alpha,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42
        }

         # Cross-validation with early stopping --> on train data - NOT on test data; else: data leakage
        if perform_cv:
            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round = num_boost_round,
                nfold=5,
                stratified=True,
                early_stopping_rounds=50,
                metrics=eval_metric,  # or ["logloss", "auc"]
                as_pandas=True,
                seed=42,
                verbose_eval=50
            )
    
            #get best boosting round from cross-validation
            best_n_estimators = len(cv_results)
            scores = cv_results[f"test-{eval_metric}-mean"]
    
            print(f"\nBest number of trees: {len(cv_results)}")
            print(f"Best test {eval_metric}: {cv_results[f'test-{eval_metric}-mean'].max():.5f}")
            print(f"sd test {eval_metric}: {cv_results[f'test-{eval_metric}-mean'].std():.5f}")
        else:
            best_n_estimators = num_boost_round

            
        # Train
        model = xgb.train(
            params = params,
            dtrain = dtrain,
            num_boost_round = best_n_estimators, #num_boost_round 
        )
            
        print("XGB model fitted.")
        
        y_test_prob = model.predict(dtest)
        y_test_pred = (y_test_prob >= classification_threshold).astype(int)

        y_train_prob = model.predict(dtrain)
        y_train_pred = (y_train_prob >= classification_threshold).astype(int)
        print("Prediction finished\n")

        return y_train_pred, y_test_pred, y_train_prob, y_test_prob

        

    def evaluate_model(self, y_true, y_pred, y_prob):
        """
        Method to evaluate a classification model using various metrics.
    
        Parameters:
        - y_true: The true labels in the set (cuPy array).
        - y_pred: predicted labels in set
        - y_proba: probabilities of predicted label
    
        Outputs:
        - Prints accuracy, precision, recall, F1 score, AUC-ROC, and confusion matrix.
        """
        from cuml.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, log_loss
        #from sklearn.metrics import f1_score, precision_score, recall_score
        
        #ensure cupy array for cuML functions
        y_true = cp.asarray(y_true)
        y_pred = cp.asarray(y_pred)
        y_prob = cp.asarray(y_prob)
        
        
        #Calculate Accuracy (using cuML for GPU acceleration)
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
            
        # Confusion Matrix (using cuML for GPU acceleration)
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix:\n{cm}")
    
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
    
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0 #share of correctly classified positives out of all as positive classified instances
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0 #share of correctly classified instances out of all actual positive instances
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 #harmonic mean of precision and recall
        
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-Score: {f1_score * 100:.2f}%")
    
        # ROC AUC
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC-ROC: {auc:.4f}")
    
        # Log Loss
        logloss = log_loss(y_true, y_prob)
        print(f"Log Loss: {logloss:.4f}")
    
        # Precision-Recall Curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        print(f"Precision-Recall Curve (length {len(thresholds)}):")
        print(f"- First threshold: {thresholds[0]:.4f}, Precision: {precisions[0]:.4f}, Recall: {recalls[0]:.4f}")
        print(f"- Last threshold: {thresholds[-1]:.4f}, Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}")

        #add best threshold F1-Score
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx = cp.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1_score = f1_scores[best_idx]
        print(f"Best F1 threshold: {best_threshold:.4f} (F1 = {best_f1_score:.4f})")
    
        # PR AUC using cuPy
        # Sort by recall
        sorted_indices = cp.argsort(recalls)
        recalls_sorted = recalls[sorted_indices]
        precisions_sorted = precisions[sorted_indices]
        
        # Now integrate
        pr_auc = cp.trapz(precisions_sorted, recalls_sorted)
        print(f"PR AUC (Average Precision): {pr_auc:.4f}")
    
        return acc, cm, precision, recall, f1_score, auc, logloss, precisions, recalls, thresholds, pr_auc, best_threshold, best_f1_score

    def evaluate_model(
        self,
        y_true,
        y_pred,
        y_prob,
        uncertainty_range: tuple = (0.4, 0.6),
        n_bootstrap_auc = 10_000,
    ):
        """
        Evaluate classification performance with and without uncertain predictions.
    
        Parameters:
        - y_true: true labels
        - y_pred: predicted labels (0/1)
        - y_prob: predicted probabilities (floats)
        - uncertainty_range: (low, high) range to exclude uncertain cases, e.g., (0.4, 0.6)
    
        Returns:
        - dict with all main metrics (overall and filtered)
        """
        from cuml.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve, log_loss
        
        #ensure cuPy arrays for cuml claculation
        y_true = cp.asarray(y_true)
        y_pred = cp.asarray(y_pred)
        y_prob = cp.asarray(y_prob)
    
        def compute_metrics(y_true, y_pred, y_prob, label="Full dataset", n_booststrap_auc = n_bootstrap_auc):
            """
            Helper function for computing metrics block.
            """
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
    
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0 #share of correctly classified positives out of all as positive classified instances
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0 #share of correctly classified instances out of all actual positive instances
            f1 = 2 * precision * recall / (precision + recall + 1e-9) #harmonic mean of precision and recall
            auc = roc_auc_score(y_true, y_prob)
            logloss_val = log_loss(y_true, y_prob)
    
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
            best_idx = cp.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]
    
            sorted_indices = cp.argsort(recalls)
            recalls_sorted = recalls[sorted_indices]
            precisions_sorted = precisions[sorted_indices]
            pr_auc = cp.trapz(precisions_sorted, recalls_sorted)
    
            print(f"\n--- {label} ---")
            print(f"Samples: {len(y_true)}")
            print(f"Accuracy: {acc * 100:.2f}%")
            print(f"Precision: {precision * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")
            print(f"F1: {f1 * 100:.2f}%")
            print(f"AUC-ROC: {auc:.4f}")
            print(f"PR AUC: {pr_auc:.4f}")
            print(f"Best F1 threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
            print(f"Confusion Matrix:\n{cm}")

            if n_booststrap_auc > 0:
                boot_aucs, p_value_boot = self._bootstrap_auc_pvalue(y_true, y_prob, n_bootstrap = n_booststrap_auc)
            else:
                boot_aucs = None
                p_value_boot = None
    
            return {
                "acc": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "logloss": float(logloss_val),
                "pr_auc": float(pr_auc),
                "best_threshold": float(best_threshold),
                "best_f1": float(best_f1),
                "n_samples": int(len(y_true)),
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "boot_aucs": boot_aucs,
                "p_value_boot": p_value_boot
            }
    
        #----Compute metrics for full dataset
        metrics_full = compute_metrics(y_true, y_pred, y_prob, label="Full dataset")
    
        #----compute metrics for filtered dataset --> exclude uncertain cases
        low, high = uncertainty_range #assign low/high ranges
        mask = (y_prob < low) | (y_prob > high) #keep only those predictions where probability is below low_threshold or greater than high_threshold
        
        #filter y based on uncertainties
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        y_prob_filtered = y_prob[mask]

        #compute metrics for filtered dataset
        if y_true_filtered.size > 0:
            metrics_filtered = compute_metrics(
                y_true_filtered,
                y_pred_filtered,
                y_prob_filtered,
                label=f"Filtered (|p-{uncertainty_range}|)",
            )
        else:
            print(f"\nNo samples outside the uncertainty range {uncertainty_range}.")
            metrics_filtered = None

        return metrics_full, metrics_filtered

    def _bootstrap_auc_pvalue(self, y_true, y_prob, n_bootstrap=10000, random_state=42):
        """
        Compute bootstrap-based p-value for testing AUC > 0.5.
    
        Parameters:
        - y_true: true labels (cuPy or NumPy array)
        - y_prob: predicted probabilities
        - n_bootstrap: number of bootstrap resamples
        - random_state: random seed for reproducibility
    
        Returns:
        - p_value_boot: bootstrap p-value for H0: AUC <= 0.5
        """
        import numpy as np
        import cupy as cp
        from cuml.metrics import roc_auc_score

        #create random number generator with random seed for reproducability
        rng = np.random.default_rng(random_state)
        
        #ensure arrays are cupy
        y_true_cp = cp.asarray(y_true)
        y_prob_cp = cp.asarray(y_prob)
        
        #number of instances = size of set
        n = len(y_true_cp)

        #create list to store results
        boot_aucs = []

        #iterate over bootstrap samples
        for _ in tqdm(range(n_bootstrap), desc = "Booststrap AUC-ROC"):
            #draw bootstrap indices
            idx = rng.integers(0, n, n)
            y_true_boot = y_true_cp[idx]
            y_prob_boot = y_prob_cp[idx]
            
            #compute AUC for resampled data
            auc_boot = float(roc_auc_score(y_true_boot, y_prob_boot))
            boot_aucs.append(auc_boot)
    
        boot_aucs = np.array(boot_aucs)
        print(f"\nMean bootstrap AUC-ROC: {np.mean(boot_aucs):.3f}")
        print(f"Std bootstrap AUC-ROC: {np.std(boot_aucs):.3f}")
        

        #compute p-value --> share of AUC-ROC's below 0.5
        p_value_boot = np.mean(boot_aucs <= 0.5)
        print(f"p-value AUC-ROC bootstrap sampling: {p_value_boot:.5f}")
    
        return boot_aucs, p_value_boot