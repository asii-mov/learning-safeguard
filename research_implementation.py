# research_implementation.py
import json
import os
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
from typing import Dict, Any
import random
from tqdm import tqdm
import logging
import datetime

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.dataset = load_dataset(
            config["dataset"]["path"],
            config["dataset"]["config"],
            split={
                "train": f"train[:{int(config['experiment']['process_sample_size'])}]",
                "test": "test",
                "validation": "validation"
            }
        )
        os.makedirs(config["dataset"]["store_dir"], exist_ok=True)

def process_dataset(config):
    dataset_manager = DatasetManager(config)
    train_dataset = dataset_manager.dataset["train"]
    
    # Collect inputs where assistant response is unsafe
    blocked_inputs = []
    for example in train_dataset:
        if example["assistant_label"] == "unsafe":  # Adjust field name as per dataset
            blocked_inputs.append(example["input"])
    
    # Save for training
    blocked_inputs_dir = "./data/blocked_inputs"
    os.makedirs(blocked_inputs_dir, exist_ok=True)
    with open(os.path.join(blocked_inputs_dir, "blocked_inputs.json"), "w") as f:
        json.dump(blocked_inputs, f)
    print(f"Collected {len(blocked_inputs)} blocked inputs.")

def train_improved_model(config):
    # Load blocked inputs
    with open("./data/blocked_inputs/blocked_inputs.json", "r") as f:
        blocked_inputs = json.load(f)
    
    # Prepare training data
    dataset_manager = DatasetManager(config)
    train_data = [{"input": inp, "label": "unsafe"} for inp in blocked_inputs]
    
    # Add safe examples for balanced training
    safe_inputs = [ex["input"] for ex in dataset_manager.dataset["train"] if ex["assistant_label"] == "safe"]
    train_data += [{"input": inp, "label": "safe"} for inp in safe_inputs[:len(blocked_inputs)]]
    
    # Convert to dataset format (simplified; adjust tokenization as needed)
    from datasets import Dataset
    train_dataset = Dataset.from_list([{"text": d["input"], "label": 1 if d["label"] == "unsafe" else 0} for d in train_data])
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(config["llama_guard"]["base_model_path"], num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        seed=config["seed"]
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train and save
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    print(f"Improved model saved to {config['training']['output_dir']}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--full-experiment", action="store_true", help="Run full experiment")
    parser.add_argument("--process-dataset", action="store_true", help="Process dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--query", help="Analyze a specific query")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    if args.full_experiment:
        process_dataset(config)
        train_improved_model(config)
        from evaluator import Evaluator
        dataset_manager = DatasetManager(config)
        evaluator = Evaluator(config, dataset_manager)
        evaluator.evaluate()
    elif args.process_dataset:
        process_dataset(config)
    elif args.train:
        train_improved_model(config)
    elif args.evaluate:
        from evaluator import Evaluator
        dataset_manager = DatasetManager(config)
        evaluator = Evaluator(config, dataset_manager)
        evaluator.evaluate()
    elif args.query:
        # Implement query analysis if needed
        pass