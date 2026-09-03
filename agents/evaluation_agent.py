import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from sklearn.preprocessing import LabelEncoder

from .base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    """Creates target distribution, confusion matrix and ROC curve (if binary)."""

    def __init__(self):
        super().__init__("EvaluationAgent")

    def run(self, context):

        # --------------------------------------------------
        # Skip if no models were trained
        # --------------------------------------------------

        if context.get("model_scores") is None:
            self.log(
                "⚠ Skipping evaluation — no trained models "
                "(single-class target)."
            )
            return context

        os.makedirs("outputs", exist_ok=True)

        best_model = context["best_model"]
        X_test = context["X_test"]
        y_test = context["y_test"]

        # Save target distribution data for InsightAgent
        context["target_info"] = (
            y_test.value_counts().to_dict()
        )

        # ==================================================
        # 1. TARGET DISTRIBUTION
        # ==================================================

        plt.figure(
            figsize=(4, 3),
            dpi=200
        )

        sns.countplot(
            x=y_test
        )

        plt.title(
            "Target Distribution"
        )

        plt.tight_layout()

        target_path = (
            "outputs/target_distribution.png"
        )

        plt.savefig(
            target_path
        )

        plt.close()

        # ==================================================
        # 2. CONFUSION MATRIX
        # ==================================================

        preds = best_model.predict(
            X_test
        )

        cm = confusion_matrix(
            y_test,
            preds
        )

        # Save confusion matrix values for InsightAgent
        context["conf_matrix_info"] = (
            cm.tolist()
        )

        plt.figure(
            figsize=(4, 3),
            dpi=200
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(
            "Confusion Matrix"
        )

        plt.ylabel(
            "True Label"
        )

        plt.xlabel(
            "Predicted Label"
        )

        plt.tight_layout()

        cm_path = (
            "outputs/confusion_matrix.png"
        )

        plt.savefig(
            cm_path
        )

        plt.close()

        # ==================================================
        # 3. ROC CURVE
        # ==================================================

        roc_path = None

        if (
            hasattr(best_model, "predict_proba")
            and y_test.nunique() == 2
        ):

            probs = best_model.predict_proba(
                X_test
            )[:, 1]

            # ----------------------------------------------
            # Convert labels such as Yes/No into 0/1
            # ----------------------------------------------

            encoder = LabelEncoder()

            y_test_roc = encoder.fit_transform(
                y_test.astype(str)
            )

            # ----------------------------------------------
            # ROC + AUC
            # ----------------------------------------------

            fpr, tpr, _ = roc_curve(
                y_test_roc,
                probs
            )

            auc = roc_auc_score(
                y_test_roc,
                probs
            )

            # Save AUC for InsightAgent
            context["auc_score"] = float(
                auc
            )

            # ----------------------------------------------
            # Plot ROC
            # ----------------------------------------------

            plt.figure(
                figsize=(4, 3),
                dpi=200
            )

            plt.plot(
                fpr,
                tpr,
                label=f"AUC = {auc:.2f}"
            )

            plt.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="grey"
            )

            plt.title(
                "ROC Curve"
            )

            plt.xlabel(
                "False Positive Rate"
            )

            plt.ylabel(
                "True Positive Rate"
            )

            plt.legend()

            plt.tight_layout()

            roc_path = (
                "outputs/roc_curve.png"
            )

            plt.savefig(
                roc_path
            )

            plt.close()

        else:
            context["auc_score"] = None

        # ==================================================
        # SAVE OUTPUT PATHS
        # ==================================================

        context["target_plot"] = (
            target_path
        )

        context["conf_matrix"] = (
            cm_path
        )

        context["roc_curve"] = (
            roc_path
        )

        return context
