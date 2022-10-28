# -*- coding: utf-8 -*-

import argparse

from datetime import datetime
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from util import *

def train(args, train_dataset, dev_dataset, model):
    tb_writer = SummaryWriter(args.output_dir)

    tr_sampler = RandomSampler(train_dataset)
    tr_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=tr_sampler)
    t_total = len(tr_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_step, num_training_steps=t_total
    )

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info(
        "  Total train batch size (w. parallel, accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)
    logging.info("  Seed = %d", args.seed)

    best_acc = 0.0
    best_acc_epoch = -1
    tr_loss = 0.0
    global_step = 0
    model.to(args.device)
    model.zero_grad()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch..."):
        model.train()
        logging_loss = 0.0
        for step, batch in enumerate(tqdm(tr_loader, desc="Iterator...")):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0], 
                "attention_mask": batch[1], 
                "token_type_ids": batch[2], 
                "labels": batch[3], 
            }
            outputs = model(**inputs)
            
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                tb_writer.add_scalar('Training Loss', loss.item(), global_step)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        tr_loss += logging_loss
        logging.info('********** Train Result **********')
        logging.info('Epoch / Total Epoch : {} / {}'.format(epoch + 1, args.num_train_epochs))
        logging.info('Loss : {:.4f}'.format(logging_loss))

        eval_metrics = eval(args, dev_dataset, model)
        for k, v in eval_metrics.items():
            tb_writer.add_scalar('Eval {}'.format(k), v, global_step)
        if epoch == 0:
            logging.info("MODEL saved at epoch {}".format(epoch + 1))
            model.save_pretrained(args.output_dir)
            best_acc = eval_metrics['Acc']
            best_acc_epoch = epoch + 1
        else:
            if eval_metrics['Acc'] > best_acc:
                logging.info("Best Acc MODEL saved at epoch {}".format(epoch + 1))
                model.save_pretrained(args.output_dir)
                best_acc = eval_metrics['Acc']
                best_acc_epoch = epoch + 1                

    logging.info('********** Train Result **********')
    logging.info(" Global step = {}, Average loss = {:.6f}".format(global_step, tr_loss / global_step))
    logging.info(" Best Acc. = {} / {}".format(best_acc, best_acc_epoch))

    tb_writer.close()

def eval(args, eval_dataset, model, submission=False):

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler)

    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))

    preds = None
    labels = None
    eval_loss = 0
    model.to(args.device)
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Test Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        if not submission:
            inputs = {
                "input_ids": batch[0], 
                "attention_mask": batch[1], 
                "token_type_ids": batch[2], 
                "labels": batch[3], 
            }
        else:
            inputs = {
                "input_ids": batch[0], 
                "attention_mask": batch[1], 
                "token_type_ids": batch[2]
            }

        outputs = model(**inputs)

        if preds is None:
            preds = outputs.logits.detach().cpu()
        else:
            preds = torch.cat((preds, outputs.logits.detach().cpu()), dim=0)

        if not submission:
            eval_loss += outputs.loss.item()
            if labels is None:
                labels = inputs["labels"].detach().cpu()
            else:
                labels = torch.cat((labels, inputs["labels"].detach().cpu()), dim=0)

    preds = torch.argmax(preds, dim=-1)

    if not submission:
        eval_metrics = dict()
        eval_metrics["Acc"] = accuracy_score(labels, preds) * 100
        eval_metrics["Loss"] = eval_loss / len(eval_dataloader)
    
        logging.info('********** Eval Result **********')
        for k, v in eval_metrics.items():
            logging.info("{}: {:.4f}".format(k, v))
        logging.info('\n' + classification_report(labels, preds, digits=4))
        logging.info("\n")
        return eval_metrics
    else:
        return preds.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default='../data/',
        type=str,
        help="The input data dir.",
    )
    parser.add_argument(
        "--output_dir",
        default='./outputs',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_name",
        default='Luyu/co-condenser-wiki',
        type=str,
        help="Pre-trained model name"
    )
    parser.add_argument(
        "--max_turns",
        default=3,
        type=int,
        help="The maximum total input turns."
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help=""
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--warmup_step",
        default=0,
        type=int,
        help="step of linear warmup"
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Whether to run submission mode"
    )
    args = parser.parse_args()

    # GPU device setting 
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 0

    args.now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.do_train and not os.path.exists(os.path.join(args.output_dir, args.now)):
        os.makedirs(os.path.join(args.output_dir, args.now))
        args.output_dir = os.path.join(args.output_dir, args.now)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='{}/log.log'.format(args.output_dir) if not os.path.exists(os.path.join(args.output_dir, 'log.log')) else '{}/log_2.log'.format(args.output_dir),
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("Arguments %s", args)
    logging.info("Device : {}  |  # of GPU : {}".format(args.device, args.n_gpu))
    
    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    special_tokens_dict = {'additional_special_tokens': ['<USER>','<SYS>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    config = BertConfig.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)


    if args.do_train:
        if os.path.exists(os.path.join("./data_cache", "train_features.bin")):
            train_dataset = torch.load(os.path.join("./data_cache", "train_features.bin"))
        else:
            train_samples = load_data(os.path.join(args.data_dir, "simmc2.1_ambiguous_candidates_dstc11_train.json"))
            train_dataset = convert_samples_to_features(args, train_samples, tokenizer)
            torch.save(train_dataset, os.path.join("./data_cache", "train_features.bin"))

        if os.path.exists(os.path.join("./data_cache", "dev_features.bin")):
            dev_dataset = torch.load(os.path.join("./data_cache", "dev_features.bin"))
        else:
            dev_samples = load_data(os.path.join(args.data_dir, "simmc2.1_ambiguous_candidates_dstc11_dev.json"))
            dev_dataset = convert_samples_to_features(args, dev_samples, tokenizer)
            torch.save(dev_dataset, os.path.join("./data_cache", "dev_features.bin"))

        train(args, train_dataset, dev_dataset, model)

    if args.do_eval:
        model = BertForSequenceClassification.from_pretrained(args.output_dir)
        if not args.submission:
            if os.path.exists(os.path.join("./data_cache", "devtest_features.bin")):
                devtest_dataset = torch.load(os.path.join("./data_cache", "devtest_features.bin"))
            else:
                devtest_samples = load_data(os.path.join(args.data_dir, "simmc2.1_ambiguous_candidates_dstc11_devtest.json"))
                devtest_dataset = convert_samples_to_features(args, devtest_samples, tokenizer)
                torch.save(devtest_dataset, os.path.join("./data_cache", "devtest_features.bin"))
            eval_metrics = eval(args, devtest_dataset, model)
        else:
            if os.path.exists(os.path.join("./data_cache", "teststd_features.bin")):
                teststd_dataset = torch.load(os.path.join("./data_cache", "teststd_features.bin"))
            else:
                teststd_samples = load_data(os.path.join(args.data_dir, "simmc2.1_ambiguous_candidates_dstc11_teststd.json"))
                teststd_dataset = convert_samples_to_features(args, teststd_samples, tokenizer)
                torch.save(teststd_dataset, os.path.join("./data_cache", "teststd_features.bin"))

            predictions = eval(args, teststd_dataset, model, submission=True)

            with open(os.path.join(args.data_dir, "simmc2.1_ambiguous_candidates_dstc11_teststd.json"), 'r', encoding='utf-8') as f:
                teststd_source_data = json.load(f)

            predicted_data = list()
            for turn_data, pred in zip(teststd_source_data['data'], predictions):
                turn_data['ambiguous_label'] = pred
                predicted_data.append(turn_data)

            with open(os.path.join(args.data_dir, "simmc2.1_ambiguous_candidates_dstc11_teststd_prediction.json"), 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "source_path": teststd_source_data["source_path"],
                        "split": teststd_source_data["split"],
                        "data": predicted_data,
                    },
                    f,
                    indent='\t'
                )
    
if __name__ == "__main__":
    main()
