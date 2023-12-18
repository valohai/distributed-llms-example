import argparse
import json
import os
import sys

import torch
import valohai
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import helpers

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


class ModelTrainer:
    def __init__(self, model_ckpt, batch_size=1, num_epochs=1, warmup_steps=500, evaluation_steps=500):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps

        self.print_gpu_report()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_ckpt).to(self.device)

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
        input_encodings = self.tokenizer(example_batch['dialogue'], padding="max_length", truncation=True)

        # with self.tokenizer.as_target_tokenizer():
        target_encodings = self.tokenizer(text_target=example_batch['summary'], padding="max_length", truncation=True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
        }

    def train(self, output_dir, train_dataset, eval_dataset):
        dataset_samsum_pt = train_dataset.map(self.convert_examples_to_features, batched=True)

        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.pretrained_model)

        trainer_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=self.evaluation_steps,
            save_steps=1e6,
            gradient_accumulation_steps=16,
            ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model=self.pretrained_model,
            args=trainer_args,
            tokenizer=self.tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt,
            eval_dataset=eval_dataset,
            callbacks=[PrinterCallback],
        )

        trainer.train()
        helpers.save_valohai_metadata(self.pretrained_model, output_dir)


class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
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
