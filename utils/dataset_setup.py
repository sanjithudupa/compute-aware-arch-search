from datasets import load_dataset, IterableDataset, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from typing import Optional, Tuple, Union

DATASET_URL = "mlfoundations/dclm-baseline-1.0"

LAYERS = 28
TOKENS_PER_DATAPOINT = 2048
TOTAL_TOKENS = 500_000_000 


def get_tokenized_dataset(
    dataset_url: str = DATASET_URL,
    tokenizer: Optional[AutoTokenizer] = None,
    tokenizer_path: str = "Qwen3-1.7B",
    max_length: int = 512,
    text_column: str = "text",
    split: str = "train",
    streaming: bool = True,
    seed: Optional[int] = None,
) -> Tuple[Union[IterableDataset, Dataset], AutoTokenizer]:
    """
    Get a tokenized dataset compatible with HuggingFace Trainer.
    
    Args:
        dataset_url: URL or path to the dataset
        tokenizer: Pre-loaded tokenizer (optional)
        tokenizer_path: Path to the tokenizer (if tokenizer not provided)
        max_length: Maximum sequence length for tokenization
        text_column: Name of the text column in the dataset
        split: Dataset split to use (e.g., "train", "validation")
        streaming: Whether to use streaming mode (default: True)
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (tokenized_dataset, tokenizer)
    """
    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(dataset_url, split=split, streaming=streaming)
    
    # Shuffle if streaming
    if streaming:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        else:
            dataset = dataset.shuffle(buffer_size=10000)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    
    map_kwargs = {
        "function": tokenize_function,
        "batched": True,
        # Always drop original columns (like language_id_whole_page_fasttext)
        # so the collator only sees tokenized fields.
        "remove_columns": dataset.column_names,
    }
    if not streaming:
        map_kwargs["desc"] = "Tokenizing dataset"

    tokenized_dataset = dataset.map(**map_kwargs)
    
    return tokenized_dataset, tokenizer


def get_data_collator(
    tokenizer: AutoTokenizer,
    mlm: bool = False,
    pad_to_multiple_of: Optional[int] = None,
) -> DataCollatorForLanguageModeling:
    """
    Get a data collator compatible with HuggingFace Trainer.
    
    Args:
        tokenizer: Tokenizer to use
        mlm: Whether to use masked language modeling (False for causal LM)
        pad_to_multiple_of: Pad sequences to multiple of this value
        
    Returns:
        DataCollatorForLanguageModeling instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
        pad_to_multiple_of=pad_to_multiple_of,
    )


if __name__ == "__main__":
    """
    Example usage with HuggingFace Trainer:
    
    from transformers import TrainingArguments, Trainer
    from qwen3_model import Qwen3ForCausalLM
    
    # Get tokenized dataset and tokenizer
    train_dataset, tokenizer = get_tokenized_dataset(
        dataset_url=DATASET_URL,
        tokenizer_path="Qwen3-1.7B",
        max_length=2048,
        streaming=True,
        seed=42,
    )
    
    # Get data collator
    data_collator = get_data_collator(tokenizer, mlm=False)
    
    # Load model
    model = Qwen3ForCausalLM.from_pretrained("Qwen3-1.7B")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        learning_rate=5e-5,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    """
    pass
