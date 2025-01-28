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
