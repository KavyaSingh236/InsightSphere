from .base_agent import BaseAgent
import os
from groq import Groq


class InsightAgent(BaseAgent):
    """Generates all narrative insights using Groq AI, with safe fallbacks."""

    def __init__(self):
        super().__init__("InsightAgent")

        api_key = os.getenv("GROQ_API_KEY")

        self.client = Groq(
            api_key=api_key
        )

    def ask_ai(self, prompt):
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4
            )

            answer = (
                resp.choices[0]
                .message
                .content
                .strip()
            )

            return answer

        except Exception as e:
            self.log(
                f"AI generation failed: {e}"
            )
            return None

    def run(self, context):

        df = context["raw_data"]

        target = context.get(
            "target_column",
            "(unknown)"
        )

        scores = (
            context.get("model_scores")
            or {}
        )

        best_name = context.get(
            "best_model_name",
            "N/A"
        )

        best_acc = context.get(
            "best_model_accuracy",
            0
        )

        n_rows, n_cols = df.shape

        schema = "\n".join(
            [
                f"- {c}: {str(df[c].dtype)}"
                for c in df.columns[:15]
            ]
        )

        # =========================================================
        # EXECUTIVE SUMMARY + MODEL STORY + RECOMMENDATIONS
        # =========================================================

        if scores:

            scores_text = ", ".join(
                [
                    f"{model}: {round(acc, 3)}"
                    for model, acc in scores.items()
                ]
            )

            main_prompt = f"""
You are a professional data science consultant.

Analyze the following machine learning results.

Dataset:
Rows: {n_rows}
Columns: {n_cols}

Prediction target:
{target}

Dataset schema:
{schema}

Model accuracies:
{scores_text}

Best model:
{best_name}

Best model accuracy:
{best_acc:.3f}

Return EXACTLY these three sections using the exact markers shown below.

<EXEC_SUM>
Write a concise 4 to 6 sentence executive summary.

Explain:
- what the dataset appears to represent based only on the column names
- what target is being predicted
- which model performed best
- the model's accuracy
- what the result means from a business perspective

<MODEL_STORY>
Write 3 to 4 short bullet points.

Explain:
- which model performed best
- how the models compared
- whether performance is weak, moderate, or strong
- what could improve future performance

<RECO>
Write 3 to 5 actionable business recommendations.

Recommendations must:
- relate specifically to the target "{target}"
- use only information supported by the dataset columns
- avoid inventing unsupported facts
- be concise and practical

Do not write anything outside these three sections.
"""

            main_text = self.ask_ai(
                main_prompt
            )

            self.log(
                f"Main Groq response: {main_text}"
            )

            # -----------------------------------------------------
            # Extract sections safely
            # -----------------------------------------------------

            def extract_section(tag, text):

                if not text:
                    return ""

                start_marker = f"<{tag}>"

                if start_marker not in text:
                    return ""

                section = text.split(
                    start_marker,
                    1
                )[1]

                possible_tags = [
                    "<EXEC_SUM>",
                    "<MODEL_STORY>",
                    "<RECO>"
                ]

                positions = []

                for marker in possible_tags:
                    pos = section.find(marker)

                    if pos != -1:
                        positions.append(pos)

                if positions:
                    section = section[
                        :min(positions)
                    ]

                return section.strip()

            exec_sum = extract_section(
                "EXEC_SUM",
                main_text
            )

            model_story = extract_section(
                "MODEL_STORY",
                main_text
            )

            reco = extract_section(
                "RECO",
                main_text
            )

            # -----------------------------------------------------
            # Fallback if Groq response fails or markers are missing
            # -----------------------------------------------------

            if not exec_sum:

                exec_sum = (
                    f"The dataset contains {n_rows} rows and "
                    f"{n_cols} columns, with '{target}' selected "
                    f"as the prediction target. "
                    f"{best_name} achieved the highest observed "
                    f"accuracy of {best_acc:.2f}. "
                    "The results indicate that the available features "
                    "contain useful predictive information, although "
                    "additional feature engineering and data refinement "
                    "may improve future model performance."
                )

            if not model_story:

                model_story = (
                    f"• {best_name} achieved the highest accuracy "
                    f"at {best_acc:.2f}.\n"
                    "• The tested models produced broadly comparable "
                    "performance.\n"
                    "• Current results indicate moderate predictive "
                    "capability.\n"
                    "• Additional feature engineering and more "
                    "representative data may improve performance."
                )

            if not reco:

                reco = (
                    f"• Investigate which available features have the "
                    f"strongest relationship with {target}.\n"
                    "• Use model predictions to identify higher-risk "
                    "cases for further review.\n"
                    "• Improve missing-value quality and collect more "
                    "relevant behavioral information where possible.\n"
                    "• Monitor model performance on new data and retrain "
                    "the model when performance declines."
                )

        else:

            exec_sum = (
                "The dataset does not currently provide enough target "
                "diversity for reliable predictive modeling. "
                "Data quality and target coverage should be improved "
                "before model performance can be evaluated reliably."
            )

            model_story = (
                "• Model training could not be completed reliably.\n"
                "• The prediction target may contain insufficient "
                "class diversity.\n"
                "• Additional representative observations are needed."
            )

            reco = (
                "• Review the selected target column.\n"
                "• Collect additional observations representing all "
                "important target classes.\n"
                "• Improve class balance before retraining.\n"
                "• Re-run InsightSphere after the dataset is updated."
            )

        context["exec_summary"] = exec_sum
        context["model_story"] = model_story
        context["recommendations_text"] = reco

        # =========================================================
        # CORRELATION INSIGHT
        # =========================================================

        corr_info = context.get(
            "corr_info",
            {}
        )

        if corr_info:

            corr_prompt = f"""
You are explaining a correlation heatmap.

Prediction target:
{target}

Top correlation information:
{corr_info}

Write ONE concise business-friendly sentence describing the
most important pattern.

Do not invent causation.
"""

            corr_insight = self.ask_ai(
                corr_prompt
            )

        else:

            corr_insight = (
                "Correlation information was not available for "
                "the selected target."
            )

        if not corr_insight:

            corr_insight = (
                "Correlation information could not be interpreted "
                "automatically."
            )

        context["corr_insight"] = (
            corr_insight
        )

        # =========================================================
        # TARGET DISTRIBUTION INSIGHT
        # =========================================================

        target_info = context.get(
            "target_info",
            {}
        )

        if target_info:

            target_prompt = f"""
You are analyzing a prediction target distribution.

Target:
{target}

Class counts:
{target_info}

Write ONE concise business-friendly sentence.

Explain whether the classes appear reasonably balanced or
whether one class is more common.

Do not use technical jargon.
"""

            target_insight = self.ask_ai(
                target_prompt
            )

        else:

            target_insight = None

        if not target_insight:

            if target_info:

                values = list(
                    target_info.values()
                )

                if len(values) >= 2:

                    largest = max(values)
                    smallest = min(values)

                    if smallest > 0:
                        ratio = largest / smallest
                    else:
                        ratio = float("inf")

                    if ratio < 1.5:

                        target_insight = (
                            f"The {target} classes are relatively "
                            "balanced, giving the model reasonable "
                            "representation of both outcomes."
                        )

                    else:

                        target_insight = (
                            f"The {target} distribution is imbalanced, "
                            "with one outcome appearing more frequently "
                            "than the other."
                        )

                else:

                    target_insight = (
                        "The target contains limited class diversity."
                    )

            else:

                target_insight = (
                    "Target distribution information was unavailable."
                )

        context["target_insight"] = (
            target_insight
        )

        # =========================================================
        # CONFUSION MATRIX INSIGHT
        # =========================================================

        cm_info = context.get(
            "conf_matrix_info",
            None
        )

        if cm_info:

            cm_prompt = f"""
You are explaining a classification confusion matrix.

Target:
{target}

Confusion matrix:
{cm_info}

Write ONE simple business-friendly sentence.

Explain whether the model correctly identifies both classes
or performs better on one class.

Do not use unnecessary technical terminology.
"""

            cm_insight = self.ask_ai(
                cm_prompt
            )

        else:

            cm_insight = None

        if not cm_insight:

            if (
                cm_info
                and len(cm_info) == 2
                and len(cm_info[0]) == 2
            ):

                tn = cm_info[0][0]
                fp = cm_info[0][1]
                fn = cm_info[1][0]
                tp = cm_info[1][1]

                if tn > tp:

                    cm_insight = (
                        "The model identifies the negative class more "
                        "successfully than the positive class, indicating "
                        "room to improve detection of positive cases."
                    )

                else:

                    cm_insight = (
                        "The model identifies both classes with varying "
                        "success, with additional tuning potentially "
                        "reducing incorrect predictions."
                    )

            else:

                cm_insight = (
                    "Confusion matrix information was unavailable."
                )

        context["cm_insight"] = (
            cm_insight
        )

        # =========================================================
        # ROC CURVE INSIGHT
        # =========================================================

        auc_val = context.get(
            "auc_score",
            None
        )

        if auc_val is not None:

            roc_prompt = f"""
You are explaining a ROC curve.

Prediction target:
{target}

AUC score:
{auc_val:.3f}

Write ONE short business-friendly sentence explaining what
this score means for the model's ability to distinguish
between the two target classes.

Avoid unnecessary technical jargon.
"""

            roc_insight = self.ask_ai(
                roc_prompt
            )

        else:

            roc_insight = None

        if not roc_insight:

            if auc_val is not None:

                if auc_val >= 0.90:
                    performance = "excellent"
                elif auc_val >= 0.80:
                    performance = "strong"
                elif auc_val >= 0.70:
                    performance = "reasonable"
                elif auc_val >= 0.60:
                    performance = "moderate"
                else:
                    performance = "limited"

                roc_insight = (
                    f"An AUC of {auc_val:.2f} indicates "
                    f"{performance} ability to distinguish between "
                    "the two target outcomes."
                )

            else:

                roc_insight = (
                    "ROC information was unavailable."
                )

        context["roc_insight"] = (
            roc_insight
        )

        # =========================================================
        # MODEL COMPARISON INSIGHT
        # =========================================================

        if scores:

            model_compare_prompt = f"""
You are comparing machine learning models.

Model accuracies:
{scores}

Best model:
{best_name}

Best accuracy:
{best_acc:.3f}

Write ONE concise sentence explaining which model performed
best and what the comparison suggests.
"""

            model_compare_insight = (
                self.ask_ai(
                    model_compare_prompt
                )
            )

        else:

            model_compare_insight = None

        if not model_compare_insight:

            if scores:

                model_compare_insight = (
                    f"{best_name} achieved the highest accuracy "
                    f"at {best_acc:.2f}, making it the strongest "
                    "baseline among the models tested."
                )

            else:

                model_compare_insight = (
                    "Model comparison was unavailable because no "
                    "models were successfully trained."
                )

        context["model_compare_insight"] = (
            model_compare_insight
        )

        return context
