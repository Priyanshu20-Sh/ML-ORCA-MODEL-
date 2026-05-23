import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


# T-LEARNER (Two-Model Learner)

# A T-Learner is one of the easiest ways to estimate causal uplift.
# The core idea:
#   1. Train a model ONLY on the control group (e.g. users who did not get an offer).
#   2. Train another separate model ONLY on the treatment group (users who got the offer).
#   3. For a new customer, predict their outcome with both models.
#   4. Causal Uplift = (Treatment Probability) - (Control Probability).
#
class TLearner(BaseEstimator, ClassifierMixin):
    """
    T-Learner (Two Models) for Uplift Modeling.
    Uses XGBoost models to predict outcomes for treatment and control groups separately.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # We define two separate XGBoost classifiers
        self.model_control = XGBClassifier(eval_metric='logloss', **self.kwargs)
        self.model_treatment = XGBClassifier(eval_metric='logloss', **self.kwargs)

    def fit(self, X, y, treatment):
        """
        Trains both models using the treatment indicator.
        
        Parameters:
            X: DataFrame or array containing features (e.g., age, income, tenure).
            y: Binary target array (e.g., 1 if they stayed/converted, 0 if they left).
            treatment: Binary array (1 for treatment group, 0 for control group).
        """
        # Convert inputs to pandas formats for easy indexing
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        treatment = pd.Series(treatment).reset_index(drop=True)
        
        # Split features and labels by treatment status
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        X_treatment = X[treatment == 1]
        y_treatment = y[treatment == 1]
        
        # Train the Control model on control data
        if not X_control.empty:
            self.model_control.fit(X_control, y_control)
            
        # Train the Treatment model on treatment data
        if not X_treatment.empty:
            self.model_treatment.fit(X_treatment, y_treatment)
        
        return self

    def predict_uplift(self, X):
        """
        Predicts the individual causal uplift score.
        Uplift Score = P(Conversion | Treated) - P(Conversion | Control)
        
        A high positive score means the user is a "Persuadable" (converts ONLY if treated).
        A negative score means the user is a "Sleeping Dog" (negative reaction to treatment).
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        
        # Get class 1 (conversion) probabilities from both models
        pred_control = self.model_control.predict_proba(X)[:, 1]
        pred_treatment = self.model_treatment.predict_proba(X)[:, 1]
            
        # Causal effect is the difference
        return pred_treatment - pred_control

    def predict(self, X):
        """
        Predicts which group is better for each customer.
        Returns 1 if treatment is predicted to help (uplift > 0), else 0.
        """
        uplift = self.predict_uplift(X)
        return (uplift > 0).astype(int)

    def get_accuracy_metrics(self, X, y, treatment):
        """
        Calculates standard classification metrics (Accuracy, F1, ROC-AUC) 
        for the internal treatment and control models on a holdout test set.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        treatment = pd.Series(treatment).reset_index(drop=True)
        
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        X_treatment = X[treatment == 1]
        y_treatment = y[treatment == 1]
        
        metrics = {}
        
        # Calculate control model accuracy
        if not X_control.empty and hasattr(self.model_control, "classes_"):
            pred_control = self.model_control.predict(X_control)
            prob_control = self.model_control.predict_proba(X_control)[:, 1]
            metrics['Control Model'] = {
                'Accuracy': accuracy_score(y_control, pred_control),
                'Precision': precision_score(y_control, pred_control, zero_division=0),
                'Recall': recall_score(y_control, pred_control, zero_division=0),
                'F1-Score': f1_score(y_control, pred_control, zero_division=0),
                'ROC-AUC': roc_auc_score(y_control, prob_control) if len(np.unique(y_control)) > 1 else 0.5
            }
        
        # Calculate treatment model accuracy
        if not X_treatment.empty and hasattr(self.model_treatment, "classes_"):
            pred_treatment = self.model_treatment.predict(X_treatment)
            prob_treatment = self.model_treatment.predict_proba(X_treatment)[:, 1]
            metrics['Treatment Model'] = {
                'Accuracy': accuracy_score(y_treatment, pred_treatment),
                'Precision': precision_score(y_treatment, pred_treatment, zero_division=0),
                'Recall': recall_score(y_treatment, pred_treatment, zero_division=0),
                'F1-Score': f1_score(y_treatment, pred_treatment, zero_division=0),
                'ROC-AUC': roc_auc_score(y_treatment, prob_treatment) if len(np.unique(y_treatment)) > 1 else 0.5
            }
        return metrics

    def get_feature_importances(self, feature_names):
        """
        Combines and returns feature importances from both models
        so we can see which features drove the predictions.
        """
        importances = {}
        if hasattr(self.model_control, "feature_importances_"):
            importances['Control Model'] = pd.Series(self.model_control.feature_importances_, index=feature_names)
        if hasattr(self.model_treatment, "feature_importances_"):
            importances['Treatment Model'] = pd.Series(self.model_treatment.feature_importances_, index=feature_names)
        
        if importances:
            df_imp = pd.DataFrame(importances)
            df_imp['Average Importance'] = df_imp.mean(axis=1)
            return df_imp.sort_values(by='Average Importance', ascending=False)
        return pd.DataFrame()


#
# S-LEARNER (Single-Model Learner)
# 
# An S-Learner trains a SINGLE global model on all the data.
# The treatment indicator is treated just like another feature (e.g. age, income).
# To predict uplift:
#   1. Set the treatment column to 1 for all rows, get the predicted probabilities.
#   2. Set the treatment column to 0 for all rows, get the predicted probabilities.
#   3. Causal Uplift = (Probability with Treatment=1) - (Probability with Treatment=0)

class SLearner(BaseEstimator, ClassifierMixin):
    """
    S-Learner (Single Model) for Uplift Modeling.
    Uses a single global XGBoost model where 'treatment' is a standard feature.
    """
    def __init__(self, treatment_col='treatment', **kwargs):
        self.treatment_col = treatment_col
        self.kwargs = kwargs
        self.model = XGBClassifier(eval_metric='logloss', **self.kwargs)

    def fit(self, X, y, treatment):
        """
        Trains the single model by appending treatment as a feature.
        """
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        treatment = pd.Series(treatment).reset_index(drop=True)
        
        # Append treatment as a column in the training dataset
        X[self.treatment_col] = treatment.values
        
        self.model.fit(X, y)
        return self

    def predict_uplift(self, X):
        """
        Predicts uplift by simulating treatment vs. control for everyone.
        """
        X_control = pd.DataFrame(X).copy().reset_index(drop=True)
        X_treatment = pd.DataFrame(X).copy().reset_index(drop=True)
        
        # Scenario A: Nobody is treated
        X_control[self.treatment_col] = 0
        
        # Scenario B: Everybody is treated
        X_treatment[self.treatment_col] = 1
        
        # Calculate predicted probabilities for both hypothetical scenarios
        pred_control = self.model.predict_proba(X_control)[:, 1]
        pred_treatment = self.model.predict_proba(X_treatment)[:, 1]
            
        # The difference is the causal effect (uplift)
        return pred_treatment - pred_control

    def predict(self, X):
        """
        Returns 1 if treatment helps (uplift > 0), else 0.
        """
        uplift = self.predict_uplift(X)
        return (uplift > 0).astype(int)

    def get_accuracy_metrics(self, X, y, treatment):
        """
        Calculates standard classification metrics for the single global model.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        treatment = pd.Series(treatment).reset_index(drop=True)
        
        # Append treatment to match the model inputs
        X[self.treatment_col] = treatment.values
        
        metrics = {}
        if hasattr(self.model, "classes_"):
            pred = self.model.predict(X)
            prob = self.model.predict_proba(X)[:, 1]
            metrics['Global Model'] = {
                'Accuracy': accuracy_score(y, pred),
                'Precision': precision_score(y, pred, zero_division=0),
                'Recall': recall_score(y, pred, zero_division=0),
                'F1-Score': f1_score(y, pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.5
            }
        return metrics

    def get_feature_importances(self, feature_names):
        """
        Returns feature importances for the single global model.
        Note that 'treatment' itself will be listed as one of the features!
        """
        features = list(feature_names) + [self.treatment_col]
        if hasattr(self.model, "feature_importances_"):
            imp = pd.Series(self.model.feature_importances_, index=features)
            return pd.DataFrame({'Global Model': imp}).sort_values(by='Global Model', ascending=False)
        return pd.DataFrame()
