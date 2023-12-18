import argparse
import json
import logging
import os
import sys
from random import Random

import datasets
import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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


class Partition:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner:
    def __init__(self, dataset, sizes=None, seed=1234):
        if sizes is None:
            sizes = [0.7, 0.2, 0.1]
        self.data = dataset
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(dataset)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


class ModelTrainer:
    def __init__(self, model_ckpt, batch_size=1, num_epochs=1, warmup_steps=500, evaluation_steps=500):
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps
        self.device = None
        self.accelerator = Accelerator()

        self.print_gpu_report()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_ckpt)
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

    def partition_dataset(self, preprocesed_dataset, collator):
        import math

        size = dist.get_world_size()
        bsz = math.ceil((2 / float(size)))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(preprocesed_dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
        set = DataLoader(
            partition,
            batch_size=bsz,
            shuffle=True,
            collate_fn=collator,
        )
        print('batch_size', bsz)
        return set, bsz

    def synchronize_and_aggregate_metrics(self, metrics):
        if metrics is None:
            return None
        group = dist.group.WORLD
        group_size = dist.get_world_size(group)

        tensor_dict = {key: torch.zeros_like(torch.tensor(metrics[key])) for key in metrics.keys()}
        metrics = {key: torch.tensor(metrics[key]).to(self.device) for key in metrics.keys()}

        for key in metrics.keys():
            # Create a tensor list with the same shape as the current key's tensor
            tensor_list = [torch.zeros_like(torch.tensor(metrics[key])).to(self.device) for _ in range(group_size)]
            # Perform the gather operation for the current key
            dist.all_gather(tensor_list, metrics[key])
            # Update the dictionary with the gathered tensors
            tensor_dict[key] = tensor_list

        mean_dict = {
            key: sum(values) / len(values) if key != 'epoch' else int(values[0]) for key, values in tensor_dict.items()
        }
        # Convert tensors to standard Python types
        mean_values_converted = {
            key: value.item() if isinstance(value, torch.Tensor) else value for key, value in mean_dict.items()
        }

        return mean_values_converted

    def train(self, output_dir, train_dataset, eval_dataset, logger, device):
        self.device = device
        print('self.device ', self.device)
        self.logger = logger
        column_names = train_dataset.column_names
        model = self.pretrained_model.to(self.device)

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

        # train_dataset_samsum_pt = train_dataset_samsum_pt.shard(num_shards=50, index=0) # Cut part of the train dataset to speed up testing
        # eval_dataset_samsum_pt = eval_dataset_samsum_pt.shard(num_shards=50, index=0) # Cut part of the train dataset to speed up testing

        seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model, pad_to_multiple_of=8)

        train_dataloader, batch_size = self.partition_dataset(train_dataset_samsum_pt, seq2seq_data_collator)
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
            epoch_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch.to(self.device))
                loss = outputs.loss
                epoch_loss += loss.item()
                loss.backward()
                average_gradients(model)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % 100 == 0:
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
                        batch["input_ids"].to(self.device),
                        attention_mask=batch["attention_mask"].to(self.device),
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

                    # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            result = metric.compute(use_stemmer=True)

            # Extract a few results from ROUGE
            result = {key: value * 100 for key, value in result.items()}

            result = {k: round(v, 4) for k, v in result.items()}

            val_logs = result.copy()
            val_logs['epoch'] = epoch

            avg_val_logs = self.synchronize_and_aggregate_metrics(val_logs)
            print("Metrics aggregated across all machines: ")
            self.dump_valohai_metadata(avg_val_logs)

            self.logger.info(result)

        if output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(model)
            helpers.save_valohai_metadata(unwrapped_model, output_dir)

    @staticmethod
    def dump_valohai_metadata(logs):
        print(json.dumps(logs))


def run(my_rank, args):
    print(args.output_dir)
    output_dir = valohai.outputs().path(args.output_dir)

    logger = logging.getLogger(__name__)
    device = torch.device("cuda:{}".format(0))

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
        logger=logger,
        device=device,
    )


def init(master_url, my_rank, world_size, fn, args):
    dist.init_process_group(init_method=master_url, rank=my_rank, world_size=world_size, backend='nccl')
    fn(my_rank, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Seq2Seq model")
    parser.add_argument("--model-ckpt", type=str, help="Pretrained model checkpoint")
    parser.add_argument("--output-dir", type=str, help="Output directory for the trained model")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps")
    parser.add_argument("--evaluation-steps", type=int, help="Evaluation steps")

    my_args = parser.parse_args()

    master_port = 1234
    master_ip = valohai.distributed.master().primary_local_ip
    url = f"tcp://{master_ip}:{master_port}"

    size = valohai.distributed.required_count
    rank = valohai.distributed.me().rank

    mp.set_start_method('spawn')
    p = mp.Process(target=init, args=(url, rank, size, run, my_args))
    p.start()
    p.join()
