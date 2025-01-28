import litellm
import streamlit as st

from config import DEFAULT_SETTINGS, GRADING_PROMPT
from utils import extract_numeric_value


class ExactMatchGrader:
    """
    A utility class for evaluating responses using exact match comparison.
    """

    @staticmethod
    def grade(response: str, ground_truth: str) -> bool:
        """
        Grades a response by checking if it exactly matches the ground truth,
        ignoring case and leading/trailing whitespace.

        Args:
            response (str): The response to be graded.
            ground_truth (str): The correct or expected answer.

        Returns:
            bool: True if the response matches the ground truth exactly, False otherwise.
        """
        return str(response).strip().lower() == str(ground_truth).strip().lower()


class LLMGrader:
    """
    A utility class for grading responses using an LLM (Large Language Model).
    The class interacts with the LLM to evaluate responses based on specified criteria
    and returns a numerical score.
    """

    @staticmethod
    def grade(
        response: str, criteria: str, model: str = DEFAULT_SETTINGS["grading_model"]
    ) -> float:
        """
        Grades a response using an LLM based on the given criteria.

        Args:
            response (str): The response to be graded.
            criteria (str): The evaluation criteria to be passed to the LLM.
            model (str, optional): The LLM model to use for grading. Defaults to "gpt-4o-mini".

        Returns:
            float: The numeric score assigned by the LLM, or 0.0 if grading fails.
        """
        try:
            prompt = GRADING_PROMPT.format(
                criteria=criteria,
                response=response,
            )

            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )

            score = extract_numeric_value(response.choices[0].message.content.strip())
            return score
        except Exception as e:
            st.error(f"Grading error: {str(e)}")
            return 0.0
