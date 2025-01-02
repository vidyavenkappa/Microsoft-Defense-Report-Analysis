from typing import Dict
from bert_score import score

class Evaluator:
    def evaluate(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate the generated answer against the ground truth.
        Args:
            generated (str): Generated answer.
            ground_truth (str): Reference answer.
        Returns:
            Dict[str, float]: Evaluation metrics (e.g., precision, recall, F1).
        """
        raise NotImplementedError("Subclasses must implement this method")

class BERTEvaluator(Evaluator):
    def evaluate(self, generated: str, references: str) -> Dict[str, float]:
        model_type = 'bert-base-uncased'
        P, R, F1 = score(
            cands=[generated], 
            refs=[references], 
            model_type=model_type, 
            lang="en"  # Specify language if needed
        )
        return {"precision": P[0].item(), "recall": R[0].item(), "f1": F1[0].item()}