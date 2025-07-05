import os, sys, json
from copy import deepcopy
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig,
    Trainer, TrainingArguments, default_data_collator, set_seed, EvalPrediction
)
from datasets import load_dataset
import evaluate
import argparse
from peft import LoraConfig, LoraModel, get_peft_model
from utils import peft_analytical_init, get_pretrained_latent_features

parser = argparse.ArgumentParser()
# basic configs
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--backbone', type=str, default='roberta-large')
# train configs
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--load-epochs', type=str, default='', help='load the epoch from a file')
parser.add_argument('--max-train-samples', type=int, default=105000)
parser.add_argument('--max-eval-samples', type=int, default=15000)
parser.add_argument('--max-test-samples', type=int, default=-1)
parser.add_argument('--per-device-train-bsz', type=int, default=64)
parser.add_argument('--per-device-eval-bsz', type=int, default=64)
parser.add_argument('--grad-accumlate-steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=4e-4, help='lr for each task needs to be tuned separately')
parser.add_argument('--load-lrs', type=str, default="", help='load the lr from a file')
parser.add_argument('--weight-decay', type=float, default=0.0)
parser.add_argument('--warmup-ratio', type=float, default=0.06)
parser.add_argument('--lr-scheduler', type=str, default='linear', choices=['linear', 'constant', 'constant_with_warmup', ])
parser.add_argument('--max-seq-len', type=int, default=128)
parser.add_argument('--gradient-checkpoint', action='store_true')
parser.add_argument('--not-load-best-model-at-end', action='store_true')
parser.add_argument('--max-num-samples', type=int, default=100, help='max number of samples to compute $H_t$ when using analytical initialization')
# lora configs
parser.add_argument('--alpha', type=int, default=16)
parser.add_argument('--rank', type=float, default=8)
parser.add_argument('--init', type=str, default='gaussian', choices=['analytical', 'gaussian', 'post'], help='initialization method for LoRA')
parser.add_argument('--freeze', type=str, default='N', choices=['A', 'B', 'N'], help='matrix A or B to be frozen; N means no freeze')
parser.add_argument('--block', type=str, default='qv', help='blocks to be adapted with LoRA, including (q,k,v)')
main_args = parser.parse_args()

'''process args:'''
model_args = {
    'llama3.2-1b': 'meta-llama/Llama-3.2-1B',
    'llama3.2-3b': 'meta-llama/Llama-3.2-3B',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
    'roberta-large': 'FacebookAI/roberta-large',
    't5-large': 'google-t5/t5-large',
}
model_path = model_args[main_args.backbone]

task_to_keys = {
    "mrpc": ("sentence1", "sentence2"), # f1/acc
    "qnli": ("question", "sentence"), # acc
    "qqp": ("question1", "question2"), # f1/acc
    "rte": ("sentence1", "sentence2"), # acc
    "sst2": ("sentence", None), # acc
    "mnli": ("premise", "hypothesis"), # acc
    "cola": ("sentence", None), # corr
    "stsb": ("sentence1", "sentence2"), # corr
}

task_to_epochs = None
if main_args.load_epochs:
    with open(main_args.load_epochs, 'r') as f:
        tasks_to_epochs = json.load(f)

tasks_to_lrs = None
if main_args.load_lrs:
    with open(main_args.load_lrs, 'r') as f:
        tasks_to_lrs = json.load(f)

pretrained_latent_features = None # for analytical initialization

def main(task_name, args):
    set_seed(args.seed)

    '''Load dataset:'''
    print(f'\n***Loading dataset: {task_name} branch of GLUE...***')
    model_class = AutoModelForSequenceClassification
    is_regression = bool(task_name == "stsb")
    datasets = load_dataset("glue", task_name, cache_dir='./.cache', trust_remote_code=True)

    task_idx = list(task_to_keys.keys()).index(task_name)
    
    if task_name == "stsb":
        num_labels = 1
    else:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
        print(f"Task: {task_name} Labels: {label_list}")
    
    if tasks_to_epochs:
        epochs = task_to_epochs[task_name]
    else:
        epochs = args.epochs

    if tasks_to_lrs:
        lr = tasks_to_lrs[task_name]
    else:
        lr = args.lr

    sentence1_key, sentence2_key = task_to_keys[task_name]
    padding = "max_length"
    max_seq_length = args.max_seq_len
    
    '''Load modules:'''
    print(f"\n***Loading model: {args.backbone}...***")
    model = model_class.from_pretrained(
        model_path,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16,
        num_labels=num_labels,
        cache_dir='./.cache'
    )
    model.pad_token_id = model.config.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='./.cache', truncation=True, return_tensors="pt", padding=True)
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    args.max_seq_len = max_seq_length

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='./.cache', truncation=True, return_tensors="pt", padding=True, max_length=max_seq_length)
    if not args.backbone.startswith('t5'):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

    '''Save path:'''
    save_root = f'./.save_dir/{task_name}/{args.backbone}-r{args.rank}-a{args.alpha}-{args.init}-fix{args.freeze}-{args.block}-wd{args.weight_decay}-seq{args.max_seq_len}'
    run_name = f'{args.backbone}-{task_name}-a{args.alpha}-r{args.rank}-{args.init}-{args.freeze}-{args.block}-wd{args.weight_decay}-seq{args.max_seq_len}'
    if args.init.startswith('analytical'):
        save_root += f'-{args.max_num_samples}' if args.max_num_samples > 0 else '-full'
    os.makedirs(save_root, exist_ok=True)

    '''Dataset processing:'''
    print(f"\n***Preprocessing dataset...***")
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    elif task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        func_args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*func_args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True)
    train_dataset = datasets["train"]
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))

    eval_dataset = datasets["validation_matched" if task_name == "mnli" else "validation"]
    if args.max_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))

    test_dataset = datasets["test_matched" if task_name == "mnli" else "test"]
    if args.max_test_samples > 0:
        test_dataset = test_dataset.select(range(min(len(test_dataset), args.max_test_samples)))

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    data_collator = default_data_collator

    '''Metric:'''
    print(f"\n***Loading metric: {task_name} for GLUE...***")
    if task_name is not None:
        metric = evaluate.load("glue", task_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task_name == 'stsb' else np.argmax(preds, axis=1)
        if task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["averaged_scores"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    '''LoRA:'''
    print(f"\n***Loading LoRA...***")
    block_to_keys = {
        'roberta-large': {
            'q': 'query',
            'k': 'key',
            'v': 'value',
        },
        't5-large': {
            'q': 'q',
            'k': 'k',
            'v': 'v',
        },
        "llama3-8b": {
            'q': 'q_proj',
            'k': 'k_proj',
            'v': 'v_proj',
        },
        "llama3.2-1b": {
            'q': 'q_proj',
            'k': 'k_proj',
            'v': 'v_proj',
        },
        "llama3.2-3b": {
            'q': 'q_proj',
            'k': 'k_proj',
            'v': 'v_proj',
        },
    }
    block_to_keys = block_to_keys[args.backbone]
    target_modules = [block_to_keys[ele] for ele in list(args.block)]
    lora_config = LoraConfig(
        init_lora_weights='gaussian', # we change the initialization later
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0,
        target_modules=target_modules,
        task_type="SEQ_CLS",
    )

    external_tasks = list(task_to_keys.keys())
    global pretrained_latent_features
    if not pretrained_latent_features:
        external_dataloaders = {}
        for external_task in external_tasks:
            external_dataset = load_dataset("glue", external_task, cache_dir='/localscratch2/hbzhang/.cache', trust_remote_code=True)
            sentence1_key, sentence2_key = task_to_keys[external_task]
            def preprocess_function(examples):
                # Tokenize the texts
                func_args = (
                    (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
                )
                result = tokenizer(*func_args, padding=padding, max_length=max_seq_length, truncation=True)

                # Map labels to IDs (not necessary for GLUE tasks)
                if label_to_id is not None and "label" in examples:
                    result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
                return result

            external_dataset = external_dataset.map(preprocess_function, batched=True)
            external_dataset = external_dataset["train"].shuffle(seed=args.seed)
            external_dataset = external_dataset.remove_columns("label")
            if args.max_num_samples > 0:
                external_dataset = external_dataset.select(range(min(len(external_dataset), args.max_num_samples)))
            else:
                external_dataset = external_dataset.select(range(min(len(external_dataset), args.max_train_samples)))
            trainer = Trainer(
                model=model.to(torch.bfloat16),
                args=TrainingArguments(
                    output_dir="./.tmp",
                    do_train=True,
                    do_eval=False,
                    do_predict=False,
                    per_device_train_batch_size=1,
                    bf16=True,
                ),
                train_dataset=external_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            external_dataloaders[external_task] = trainer.get_train_dataloader()
        pretrained_latent_features = get_pretrained_latent_features(deepcopy(model), args.rank, external_dataloaders, target_modules)
    right_lora_init_weights = {k: v for k, v in pretrained_latent_features.items() if k != task_name}
    right_lora_init_weights = peft_analytical_init(args.rank, right_lora_init_weights)
    peft_model = get_peft_model(model, lora_config)
    if args.init.startswith('analytical'):
        for name, param in peft_model.named_parameters():
            if 'lora_A' in name:
                param.data = right_lora_init_weights[name].to(param.device)

    # Freeze A or B or none
    if 'A' in args.freeze and 'B' in args.freeze:
        raise ValueError('Cannot freeze both A and B')
    for name, param in peft_model.named_parameters():
        if 'A' in args.freeze:
            if 'lora_A' in name:
                param.requires_grad = False
        if 'B' in args.freeze:
            if 'lora_B' in name:
                param.requires_grad = False
    peft_model.print_trainable_parameters()
    peft_model.save_pretrained(os.path.join(save_root, "init_model"))

    '''Train and eval:'''
    print(f"\n***Loading training and evaluation args...***")
    # Initialize our Trainer
    training_args = TrainingArguments(
        output_dir=save_root,
        do_train=True,
        do_eval=True,
        do_predict=False,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_bsz,
        per_device_eval_batch_size=args.per_device_eval_bsz,
        num_train_epochs=epochs,
        gradient_accumulation_steps=args.grad_accumlate_steps,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        save_strategy="epoch",
        gradient_checkpointing=args.gradient_checkpoint,
        run_name=run_name,
        logging_steps=100,
        logging_first_step=True,
        logging_dir=f'{save_root}/logs',
        load_best_model_at_end=not args.not_load_best_model_at_end,
        metric_for_best_model="matthews_correlation" if task_name == "cola" else "averaged_scores" if task_name == 'stsb' else "accuracy",
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        print(f"\n***Start training***")
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            args.max_train_samples if args.max_train_samples > 0 else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))
        
        trainer.model.save_pretrained(training_args.output_dir)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        print(f"\n***Start evaluation***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [task_name]
        eval_datasets = [eval_dataset]
        if task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    filename = os.path.join(save_root, 'args.json')
    with open(filename, 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)
    print(f"Arguments saved to {filename}")

if __name__ == '__main__':
    for task_name in task_to_keys.keys():
        print(f"***Fine-tuning on {task_name}...***")
        main(task_name, deepcopy(main_args))
