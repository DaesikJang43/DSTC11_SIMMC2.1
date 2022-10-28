# -*- coding: utf-8 -*-

import argparse

from datetime import datetime
from re import I
from transformers import ViTFeatureExtractor, AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

from model import *
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
    best_acc_dict = None
    best_acc_epoch = -1
    tr_loss = 0.0
    global_step = 0
    model.to(args.device)
    model.zero_grad()
    eval_metrics = eval(args, dev_dataset, model)
    for epoch in trange(int(args.num_train_epochs), desc="Epoch..."):
        model.train()
        logging_loss = 0.0
        for step, batch in enumerate(tqdm(tr_loader, desc="Iterator...")):
            batch = tuple(t.to(args.device) for t in batch)
            if args.domain == 'fashion':
                inputs = {
                    "pixel_values": batch[0], 
                    "type_labels": batch[1], 
                    "color_labels": batch[2], 
                    "pattern_labels": batch[3], 
                    "sleeve_labels": batch[4]
                }
            else:
                inputs = {
                    "pixel_values": batch[0], 
                    "type_labels": batch[1], 
                    "color_labels": batch[2], 
                }

            outputs = model(**inputs)
            
            loss = outputs[1]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                tb_writer.add_scalar('Training Loss', loss.item(), global_step)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        tr_loss += logging_loss
        logging.info('********** Train Result **********')
        logging.info('Epoch / Total Epoch : {} / {}'.format(epoch + 1, args.num_train_epochs))
        logging.info('Loss : {:.4f}'.format(logging_loss / len(tr_loader)))

        eval_metrics = eval(args, dev_dataset, model)
        tb_writer.add_scalar(" Eval Loss", eval_metrics["Loss"], global_step)
        for k, v in eval_metrics["Acc"].items():
            if ards.domain == "fashion" and k == "color":
                tb_writer.add_scalar(" Eval {} Macro F1".format(k), v, global_step)
            else:
                tb_writer.add_scalar(" Eval {} Acc".format(k), v, global_step)
        if epoch == 0:
            logging.info("MODEL saved at epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
            best_acc = sum(eval_metrics['Acc'].values()) / len(eval_metrics['Acc'])
            best_loss = eval_metrics["Loss"]
            best_acc_dict = eval_metrics.copy()
            best_acc_epoch = epoch + 1
        else:
            new_acc = sum(eval_metrics['Acc'].values()) / len(eval_metrics['Acc'])
            if (new_acc > best_acc) or (new_acc == best_acc and best_loss > eval_metrics["Loss"]):
                logging.info("Best MODEL saved at epoch {}".format(epoch + 1))
                torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                best_acc = new_acc
                best_loss = eval_metrics["Loss"]
                best_acc_dict = eval_metrics.copy()
                best_acc_epoch = epoch + 1

    logging.info('********** Train Result **********')
    logging.info(" Train Loss: {:.4f}".format(tr_loss / t_total))
    for k, v in best_acc_dict["Acc"].items():
        logging.info(" {} Best Acc: {:.4f}".format(k, v))

    tb_writer.close()

def eval(args, eval_dataset, model, submission=False):

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler)

    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))

    preds = defaultdict(list)
    labels = defaultdict(list)
    model.to(args.device)
    model.eval()
    eval_loss = 0.0
    for batch in tqdm(eval_dataloader, desc="Test Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        if submission:
            inputs = {
                "pixel_values": batch[0]
            }
        elif args.domain == 'fashion':
            inputs = {
                "pixel_values": batch[0], 
                "type_labels": batch[1], 
                "color_labels": batch[2], 
                "pattern_labels": batch[3], 
                "sleeve_labels": batch[4]
            }
        else:
            inputs = {
                "pixel_values": batch[0], 
                "type_labels": batch[1], 
                "color_labels": batch[2], 
            }

        with torch.no_grad():
            if submission:
                logits = model(**inputs)
            else:
                logits, loss = model(**inputs)
                eval_loss += loss.item()

                labels['type'].append(inputs["type_labels"].detach().cpu())
                labels['color'].append(inputs["color_labels"].detach().cpu())

                if args.domain == 'fashion':
                    labels['pattern'].append(inputs["pattern_labels"].detach().cpu())
                    labels['sleeveLength'].append(inputs["sleeve_labels"].detach().cpu())

            if "1" in args.task:
                preds['type'].append(logits[0].detach().cpu())
            if "2" in args.task:
                preds['color'].append(logits[1].detach().cpu())

            if args.domain == 'fashion':
                if "3" in args.task:
                    preds['pattern'].append(logits[2].detach().cpu())
                if "4" in args.task:
                    preds['sleeveLength'].append(logits[3].detach().cpu())


    for key in preds:
        preds[key] = torch.cat(preds[key], dim=0)
        if not submission:
            labels[key] = torch.cat(labels[key], dim=0).numpy()
            if args.domain == 'fashion' and key == 'color':
                color_preds = (preds[key] > 0).type(torch.long)
                no_label_idx = (color_preds.sum(-1) == 0).nonzero().view(-1)
                argmax_idx = torch.argmax(preds[key][no_label_idx], dim=-1)
                color_preds[no_label_idx, argmax_idx] = 1
                preds[key] = color_preds.numpy()
            else:
                preds[key] = torch.argmax(preds[key], dim=-1).numpy()
        else:
            if args.domain == 'fashion' and key == 'color':
                color_preds = (preds[key] > 0).type(torch.long)
                no_label_idx = (color_preds.sum(-1) == 0).nonzero().view(-1)
                argmax_idx = torch.argmax(preds[key][no_label_idx], dim=-1)
                color_preds[no_label_idx, argmax_idx] = 1
                preds[key] = color_preds.tolist()
            else:
                preds[key] = torch.argmax(preds[key], dim=-1).tolist()
    
    if not submission:
        eval_metrics = dict()
        eval_metrics["Acc"] = dict()
        eval_metrics["Loss"] = eval_loss / len(eval_dataloader)
        for key in preds:
            if args.domain == 'fashion' and key == 'color':
                colors = list()
                for i in range(preds[key].shape[1]):
                    colors.append(f1_score(labels[key][:, i], preds[key][:, i], zero_division=0) * 100)
                eval_metrics["Acc"][key] = sum(colors) / len(colors)
            else:
                eval_metrics["Acc"][key] = accuracy_score(labels[key], preds[key]) * 100
    
        logging.info('********** Eval Result **********')
        logging.info(" Eval Loss: {:.4f}".format(eval_metrics["Loss"]))
        for k, v in eval_metrics["Acc"].items():
            if args.domain == "fashion" and k == "color":
                logging.info(" Eval {} Macro F1: {:.4f}".format(k, v))
            else:
                logging.info(" Eval {} Acc: {:.4f}".format(k, v))
        return eval_metrics
    else:
        return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default='./data/',
        type=str,
        help="The input data dir.",
    )
    parser.add_argument(
        "--image_data_dir",
        default='../',
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
        "--vit_model",
        default='google/vit-large-patch32-384',
        type=str,
        help="Pre-trained Vit model name"
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
        default=16,
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
        default=9e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Gradient accumulation steps size"
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
    parser.add_argument(
        "--domain",
        default="fashion",
        choices=["fashion", "furniture"],
        type=str,
        help="Choice data mode [fashion, furniture]"
    )
    parser.add_argument(
        "--task",
        default="1234",
        type=str,
        help="1: type, 2: color, 3: pattern, 4: sleeve"
    )
    args = parser.parse_args()
    args.task = list(args.task)

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
    
    label2idx_dict = dict()
    args.num_labels = dict()
    if args.domain == 'fashion':
        for class_type in ['type', 'color', 'pattern', 'sleeveLength']:
            with open(os.path.join(args.data_dir, "image_classification", "fashion_{}.json".format(class_type)))as f:
                label2idx_dict[class_type] = json.load(f)
                args.num_labels[class_type] = len(label2idx_dict[class_type])
    else:
        for class_type in ['type', 'color']:
            with open(os.path.join(args.data_dir, "image_classification", "furniture_{}.json".format(class_type)))as f:
                label2idx_dict[class_type] = json.load(f)
                args.num_labels[class_type] = len(label2idx_dict[class_type])

    if args.domain == 'fashion' and os.path.exists(os.path.join("./data", "fashion_color_weight.json")):
        with open(os.path.join("./data", "fashion_color_weight.json"), 'r', encoding='utf-8') as f:
            color_weight = json.load(f)
            args.positive_weight = [c for c in color_weight.values()]

    set_seed(args)
    extractor = ViTFeatureExtractor.from_pretrained(args.vit_model)
    model = ImageClassifier(args)
    model.to(args.device)

    if args.do_train:
    
        if os.path.exists(os.path.join("./data_cache", "{}_train_dataset.bin".format(args.domain))):
            train_dataset = torch.load(os.path.join("./data_cache", "{}_train_dataset.bin".format(args.domain)))
        else:
            train_samples = load_data(os.path.join(args.data_dir, "image_classification", "{}_train.json".format(args.domain)))
            train_dataset = convert_samples_to_features(args, train_samples, extractor, label2idx_dict)
            torch.save(train_dataset, os.path.join("./data_cache", "{}_train_dataset.bin".format(args.domain)))

        if os.path.exists(os.path.join("./data_cache", "{}_dev_dataset.bin".format(args.domain))):
            dev_dataset = torch.load(os.path.join("./data_cache", "{}_dev_dataset.bin".format(args.domain)))
        else:
            dev_samples = load_data(os.path.join(args.data_dir, "image_classification", "{}_dev.json".format(args.domain)))
            dev_dataset = convert_samples_to_features(args, dev_samples, extractor, label2idx_dict)
            torch.save(dev_dataset, os.path.join("./data_cache", "{}_dev_dataset.bin".format(args.domain)))

        train(args, train_dataset, dev_dataset, model)

    if args.do_eval:
        model = ImageClassifier(args)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model.bin")))
        logging.info("Success Model Load")
        
        if not args.submission:
            if os.path.exists(os.path.join("./data_cache", "{}_devtest_dataset.bin".format(args.domain))):
                devtest_dataset = torch.load(os.path.join("./data_cache", "{}_devtest_dataset.bin".format(args.domain)))
            else:
                devtest_samples = load_data(os.path.join(args.data_dir, "image_classification", "{}_devtest.json".format(args.domain)))
                devtest_dataset = convert_samples_to_features(args, devtest_samples, extractor, label2idx_dict)
                torch.save(devtest_dataset, os.path.join("./data_cache", "{}_devtest_dataset.bin".format(args.domain)))
                
            eval_metrics = eval(args, devtest_dataset, model)
        else:
            idx2label_dict = dict()
            for class_type, label2idx in label2idx_dict.items():
                idx2label_dict[class_type] = {idx:label for label, idx in label2idx.items()}

            for data_type in ["dev", "devtest", "teststd"]:
                if args.domain == "furniture" and data_type == "dev":
                    with open(os.path.join(args.data_dir, "image_classification", '{}_{}_prediction.json'.format(args.domain, data_type)), 'w', encoding='utf-8') as f:
                        json.dump({}, f, indent='\t')
                    continue
                data_name = "{}_{}_testing.json".format(args.domain, data_type) if data_type != "teststd" else "{}_{}.json".format(args.domain, data_type)
                teststd_samples = load_data(os.path.join(args.data_dir, "image_classification", data_name))
                teststd_dataset = convert_samples_to_features(args, teststd_samples, extractor, label2idx_dict, eval=True, data_type=data_type)

                predictions = eval(args, teststd_dataset, model, submission=True)

                new_samples = defaultdict(dict)
                with open(os.path.join(args.data_dir, "image_classification", data_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for idx, d in enumerate(data):
                    feature = dict()
                    for class_type, pred in predictions.items():
                        if class_type == 'color' and args.domain == 'fashion':                        
                            feature[class_type] = [idx2label_dict[class_type][idx] for idx, i in enumerate(pred[idx]) if i == 1]
                        else:
                            feature[class_type] = idx2label_dict[class_type][pred[idx]]
                    new_samples[d['scene_name']][d['index']] = feature

                with open(os.path.join(args.data_dir, "image_classification", '{}_{}_prediction.json'.format(args.domain, data_type)), 'w', encoding='utf-8') as f:
                    json.dump(new_samples, f, indent='\t')
    
if __name__ == "__main__":
    main()
