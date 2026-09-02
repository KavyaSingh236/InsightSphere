from .base_agent import BaseAgent
import os
from groq import Groq


class TargetAgent(BaseAgent):
    """Uses strong target-name rules first, then Groq LLM, then fallback."""

    def __init__(self):
        super().__init__("TargetAgent")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _ask_ai_for_target(self, df):
        schema_lines = [f"{c}: {str(df[c].dtype)}" for c in df.columns]
        schema = "\n".join(schema_lines)

        prompt = (
            "You are configuring an AutoML pipeline.\n"
            "Here are the dataset columns with dtypes:\n"
            f"{schema}\n\n"
            "Which single column is most likely the prediction target/label?\n"
            "Do not choose obvious ID columns.\n"
            "Reply with only the exact column name."
        )

        resp = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
        )

        answer = resp.choices[0].message.content.strip()

        return (
            answer
            .replace('"', "")
            .replace("'", "")
            .strip()
        )

    def run(self, context):
        df = context["data"]

        self.log("🤖 Inferring target column...")

        target_col = None

        # --------------------------------------------------
        # 1. Strong exact target-name detection FIRST
        # --------------------------------------------------

        priority_keywords = [
            "target",
            "label",
            "class",
            "outcome",
            "churn",
            "default",
            "fraud",
            "survived",
            "diagnosis",
            "readmitted",
            "response",
            "result",
            "status"
        ]

        for keyword in priority_keywords:
            for col in df.columns:

                if col.lower().strip() == keyword:
                    target_col = col

                    self.log(
                        f"Target detected by exact keyword = {col}"
                    )

                    break

            if target_col is not None:
                break

        # --------------------------------------------------
        # 2. Partial keyword match
        # --------------------------------------------------

        if target_col is None:

            for col in df.columns:

                low = col.lower().strip()

                if any(
                    keyword in low
                    for keyword in priority_keywords
                ):
                    target_col = col

                    self.log(
                        f"Target detected by keyword = {col}"
                    )

                    break

        # --------------------------------------------------
        # 3. Ask Groq only if no obvious target found
        # --------------------------------------------------

        if target_col is None:

            try:

                ai_guess = self._ask_ai_for_target(df)

                if ai_guess in df.columns:
                    target_col = ai_guess

                    self.log(
                        f"AI selected target = {ai_guess}"
                    )

            except Exception as e:

                self.log(
                    f"AI target inference failed: {e}"
                )

        # --------------------------------------------------
        # 4. Final fallback = last column
        # --------------------------------------------------

        if target_col is None:

            target_col = df.columns[-1]

            self.log(
                f"Using last column as target = {target_col}"
            )

        context["target_column"] = target_col

        self.log(
            f"Final target column = {target_col}"
        )

        return context
