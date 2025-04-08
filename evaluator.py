# evaluator.py
from typing import Dict, Any
import os
import json
import random
from tqdm import tqdm
import logging
import datetime
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class LlamaGuardFilter:
    def __init__(self, model_path):
        # Placeholder; implement actual model loading
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    def check_content(self, text):
        # Placeholder; implement prediction logic
        # Returns (is_safe: bool, reason: str)
        return (True, "Safe")  # Adjust based on actual implementation

class Evaluator:
    """Evaluates default and improved Llama Guard models."""
    
    def __init__(self, config: Dict[str, Any], dataset_manager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.default_model_path = config["llama_guard"]["default_model_path"]
        self.improved_model_path = config["training"]["output_dir"]
        self.results_dir = config["evaluation"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate default and improved models on test set."""
        seed = self.config.get("seed", 42)
        logger.info("Starting evaluation")
        
        # Load models
        default_guard = LlamaGuardFilter(self.default_model_path)
        improved_guard = LlamaGuardFilter(self.improved_model_path)
        
        # Get test dataset
        test_dataset = self.dataset_manager.dataset["test"]
        eval_size = min(len(test_dataset), self.config["evaluation"]["max_eval_samples"])
        random.seed(seed)
        eval_indices = random.sample(range(len(test_dataset)), eval_size)
        eval_dataset = test_dataset.select(eval_indices)
        logger.info(f"Evaluating on {eval_size} samples")
        
        # Initialize results
        results = {
            "default_model": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "predictions": []},
            "improved_model": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "predictions": []}
        }
        
        # Evaluate
        for example in tqdm(eval_dataset, desc="Evaluating"):
            input_text = example["user_input"]
            true_label = example["human_annotation_toxicity"]
            
            # Default model
            default_safe, default_reason = default_guard.check_content(input_text)
            default_pred = not default_safe
            self._update_metrics(results["default_model"], default_pred, true_label, input_text, default_reason)
            
            # Improved model
            improved_safe, improved_reason = improved_guard.check_content(input_text)
            improved_pred = not improved_safe
            self._update_metrics(results["improved_model"], improved_pred, true_label, input_text, improved_reason)
        
        # Calculate metrics
        for model_name in results:
            r = results[model_name]
            tp, fp, tn, fn = r["tp"], r["fp"], r["tn"], r["fn"]
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            r["metrics"] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")
        
        # Summary
        summary = {
            "default_model": results["default_model"]["metrics"],
            "improved_model": results["improved_model"]["metrics"],
            "improvement": {
                k: results["improved_model"]["metrics"][k] - results["default_model"]["metrics"][k]
                for k in ["accuracy", "precision", "recall", "f1"]
            }
        }
        return summary
    
    def _update_metrics(self, result, pred, true, input_text, reason):
        if pred and true:
            result["tp"] += 1
        elif pred and not true:
            result["fp"] += 1
        elif not pred and not true:
            result["tn"] += 1
        elif not pred and true:
            result["fn"] += 1
        result["predictions"].append({
            "input": input_text,
            "true_label": "unsafe" if true else "safe",
            "predicted_label": "unsafe" if pred else "safe",
            "reason": reason
        })