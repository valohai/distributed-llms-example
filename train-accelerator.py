import argparse
import json
import logging
import os
import sys

import datasets
import evaluate
import numpy as np
import torch
import valohai
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

import helpers

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class ModelTrainer:
    def __init__(self, model_ckpt, batch_size=1, num_epochs=1, warmup_steps=500, evaluation_steps=500):
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.print_gpu_report()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_ckpt).to(self.device)
        self.logger = logging.getLogger(__name__)
        self.set_logs()

    def set_logs(self):
        self.logger.info(self.accelerator.state)
        self.logger.setLevel(logging.INFO if self.accelerator.is_local_main_process else logging.ERROR)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
        else:
            datasets.utils.logging.set_verbosity_error()

    def print_gpu_report(self):
        from subprocess import call

        print('torch.cuda.device_count() ', torch.cuda.device_count())
        print('self.device ', self.device)
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDA VERSION')
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        call(
            [
                "nvidia-smi",
                "--format=csv",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
            ],
        )
        print('Active CUDA Device: GPU', torch.cuda.current_device())

        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())

    def generate_batch_sized_chunks(self, list_of_elements):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), self.batch_size):
            yield list_of_elements[i : i + self.batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric):
        article_batches = list(self.generate_batch_sized_chunks(dataset['article']))
        target_batches = list(self.generate_batch_sized_chunks(dataset['highlights']))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = self.tokenizer(
                article_batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            summaries = self.pretrained_model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128,
            )

            decoded_summaries = [
                self.tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries
            ]

            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        score = metric.compute()
        return score

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            padding="max_length",
            truncation=True,
            max_length=1024,
        )

        target_encodings = self.tokenizer(
            text_target=example_batch['summary'],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
        }

    def synchronize_and_aggregate_metrics(self, metrics):
        torch_metrics = {key: torch.tensor(metrics[key]).to(self.device) for key in metrics.keys()}
        metrics_list = self.accelerator.gather(torch_metrics)
        self.accelerator.wait_for_everyone()

        return {metric: torch.mean(metrics_list[metric]).item() for metric in metrics_list}

    def train(self, output_dir, train_dataset, eval_dataset):
        column_names = train_dataset.column_names
        train_dataset_samsum_pt = train_dataset.map(
            self.convert_examples_to_features,
            batched=True,
            remove_columns=column_names,
        )
        eval_dataset_samsum_pt = eval_dataset.map(
            self.convert_examples_to_features,
            batched=True,
            remove_columns=column_names,
        )

        seq2seq_data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.pretrained_model,
            pad_to_multiple_of=8 if self.accelerator.mixed_precision == 'fp16' else None,
        )
        # train_dataset_samsum_pt = train_dataset_samsum_pt.shard(num_shards=10,
        #                                                         index=0)  # Cut part of the train dataset to speed up testing
        # eval_dataset_samsum_pt = eval_dataset_samsum_pt.shard(num_shards=10,
        #                                                       index=0)  # Cut part of the train dataset to speed up testing

        train_dataloader = DataLoader(
            train_dataset_samsum_pt,
            shuffle=True,
            collate_fn=seq2seq_data_collator,
            batch_size=1,
        )

        eval_dataloader = DataLoader(eval_dataset_samsum_pt, collate_fn=seq2seq_data_collator, batch_size=1)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.pretrained_model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.pretrained_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
            self.pretrained_model,
            optimizer,
            train_dataloader,
            eval_dataloader,
        )
        self.logger.info("***** Accelerator prepared for training *****")

        num_update_steps_per_epoch = len(train_dataloader)
        max_train_steps = self.num_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_training_steps=max_train_steps,
            num_warmup_steps=1,
        )

        metric = evaluate.load("rouge")

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.num_epochs}")

        progress_bar = tqdm(range(max_train_steps))
        completed_steps = 0

        for epoch in range(self.num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch.to(self.device))
                loss = outputs.loss
                self.accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % 300 == 0:
                    logs = {'loss': loss.item(), 'step': completed_steps}
                    self.dump_valohai_metadata(logs)

                if completed_steps >= max_train_steps:
                    break

            model.eval()

            gen_kwargs = {
                "max_length": 128,
                "num_beams": 2,
            }
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = self.accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    generated_tokens = self.accelerator.pad_across_processes(
                        generated_tokens,
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                    )
                    labels = batch["labels"]
                    generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
                    labels = self.accelerator.gather(labels).cpu().numpy()

                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            metrics = metric.compute(use_stemmer=True)

            # Synchronize and calculate mean across GPUs
            avg_metrics = self.synchronize_and_aggregate_metrics(metrics)

            self.dump_valohai_metadata(avg_metrics)
            self.logger.info("Metrics aggregated across all GPUs: ")
            self.logger.info(avg_metrics)

        if output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(model)
            helpers.save_valohai_metadata(unwrapped_model, output_dir)

    def dump_valohai_metadata(self, logs):
        print(json.dumps(logs))


def run(args):
    output_dir = valohai.outputs().path(args.output_dir)

    data_path = os.path.dirname(valohai.inputs('dataset').path())
    dataset_samsum = load_dataset(
        'json',
        data_files={
            'train': os.path.join(data_path, 'train.json'),
            'validation': os.path.join(data_path, 'val.json'),
        },
    )
    train_dataset = dataset_samsum["train"]
    eval_dataset = dataset_samsum["validation"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(eval_dataset)}")

    trainer = ModelTrainer(
        model_ckpt=args.model_ckpt,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.evaluation_steps,
    )

    trainer.train(
        output_dir=output_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model")
    parser.add_argument("--model-ckpt", type=str, help="Pretrained model checkpoint")
    parser.add_argument("--output-dir", type=str, help="Output directory for the trained model")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps")
    parser.add_argument("--evaluation-steps", type=int, help="Evaluation steps")

    args = parser.parse_args()
    run(args)
