import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
import time
from typing import Optional
import warnings

from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,    
    HfArgumentParser,
    AutoModelForCausalLM, 
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory to store pretrained models"
            )
        },
    )    
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type: 8b "},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the tokenized dataset to use (via the datasets library)."}
    )
    context_length: Optional[int] = field(
        default=2048, metadata={"help": "The name of the tokenized dataset to use (via the datasets library)."}
    )


def main(): 

    # Settings the warnings to be ignored 
    warnings.filterwarnings('ignore') 
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)
    # datasets.utils.logging.set_verbosity(logging.WARN)
    # transformers.utils.logging.set_verbosity(logging.WARN)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    with training_args.main_process_first(desc="model directory"):
        if os.path.isdir(os.path.join(model_args.model_dir, model_args.model_name_or_path)):
            logger.info(f'Model directory already exists\n')
        else:
            logger.info(f'Create model directory\n')
            os.mkdir(os.path.join(model_args.model_dir, model_args.model_name_or_path))

        # Load datasetcompute_loss
        logger.info(f'Load tokenized datasets: {data_args.dataset_name_or_path}\n')
        tokenized_datasets = load_from_disk(data_args.dataset_name_or_path)
        #tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)
        logger.info(tokenized_datasets)

    # Get tokenizer
    logger.info(f'Get tokenizer: {model_args.tokenizer_name_or_path}\n')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token="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",)
    sample_text = "카터는 이집트와 이스라엘을 조정하여 중동 평화를 위한 캠프데이비드 협정을 체결했다."
    logger.info(tokenizer.tokenize(sample_text))


    # Load model
    logger.info(f'Load model: {model_args.model_name_or_path}\n')
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token="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",)
    logger.info(model)


    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    with training_args.main_process_first(desc="data collator"):
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test']
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_datasets['train'])

        trainer.log_metrics("train", metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(tokenized_datasets['test'])
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)         

                                                 
if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info("Total Time:{:.4f}".format(time.time() - start_time))
