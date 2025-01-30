from typing import Dict, Tuple

import litellm
import nltk
import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import DEFAULT_SETTINGS, GRADING_PROMPT
from utils import extract_numeric_value, get_device


class GradingHandler:
    """Orchestrates response grading using selected evaluation methods"""

    def __init__(
        self,
        response: str,
        ground_truth: str,
        eval_method: str,
        criteria: str = "",
        judge_model: str = None,
    ):
        self.response = response
        self.ground_truth = ground_truth
        self.eval_method = eval_method
        self.criteria = criteria
        self.judge_model = judge_model or DEFAULT_SETTINGS["grading_model"]

    def grade(self) -> Dict[str, float]:
        """Perform grading based on selected evaluation method"""
        grades = {}

        if self.eval_method in ("Exact Match", "Combined (All Methods)"):
            grades["Score"] = ExactMatchGrader.grade(self.response, self.ground_truth)

        if self.eval_method in ("Overlap Metrics", "Combined (All Methods)"):
            grades["ROUGE Score"], grades["BLEU Score"] = OverlapGrader.grade(
                self.response, self.ground_truth
            )

        if self.eval_method in ("Semantic Similarity", "Combined (All Methods)"):
            grades["Semantic Score"] = SemanticSimilarityGrader.grade(
                self.response, self.ground_truth
            )

        if self.eval_method in ("LLM Criteria", "Combined (All Methods)"):
            grades["LLM Score"] = LLMGrader.grade(
                self.response, self.criteria, self.judge_model
            )

        return grades


class ExactMatchGrader:
    """Evaluates responses using exact match comparison"""

    @staticmethod
    def grade(response: str, ground_truth: str) -> float:
        normalized_response = str(response).strip().lower()
        normalized_truth = str(ground_truth).strip().lower()
        return float(normalized_response == normalized_truth)


class OverlapGrader:
    """Evaluates responses using text overlap metrics (ROUGE and BLEU)"""

    _rouge_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    _bleu_smoothing = SmoothingFunction()

    @classmethod
    def grade(cls, response: str, ground_truth: str) -> Tuple[float, float]:
        return (
            cls._calculate_rouge(response, ground_truth),
            cls._calculate_bleu(response, ground_truth),
        )

    @classmethod
    def _calculate_rouge(cls, response: str, ground_truth: str) -> float:
        scores = cls._rouge_scorer.score(ground_truth, response)
        return (
            scores["rouge1"].fmeasure
            + scores["rouge2"].fmeasure
            + scores["rougeL"].fmeasure
        ) / 3

    @classmethod
    def _calculate_bleu(cls, response: str, ground_truth: str) -> float:
        reference = nltk.word_tokenize(ground_truth.lower())
        candidate = nltk.word_tokenize(response.lower())
        return sentence_bleu(
            [reference], candidate, smoothing_function=cls._bleu_smoothing.method1
        )


class SemanticSimilarityGrader:
    """Evaluates responses using semantic similarity metrics"""

    _device = get_device()
    _model = None

    @classmethod
    def grade(cls, response: str, ground_truth: str) -> float:
        cls._load_model()
        try:
            embeddings = cls._model.encode(
                [response, ground_truth], convert_to_tensor=True
            ).to(cls._device)

            return float(
                cosine_similarity(
                    embeddings[0].cpu().numpy().reshape(1, -1),
                    embeddings[1].cpu().numpy().reshape(1, -1),
                )[0][0]
            )
        except Exception as e:
            st.error(f"Semantic grading error: {str(e)}")
            return 0.0

    @classmethod
    @st.cache_resource
    def _load_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(DEFAULT_SETTINGS["embedding_model"])


class LLMGrader:
    """Evaluates responses using LLM-based criteria grading"""

    @staticmethod
    def grade(response: str, criteria: str, model: str) -> float:
        try:
            prompt = GRADING_PROMPT.format(criteria=criteria, response=response)
            result = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return extract_numeric_value(result.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"LLM grading error: {str(e)}")
            return 0.0
