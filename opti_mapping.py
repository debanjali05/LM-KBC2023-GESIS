import json
import argparse
import os
import random
import sys
import logging
import faiss

import datasets
from tqdm import tqdm
import wandb
import torch
import random
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import csv
from tqdm import tqdm
from functools import partial
from transformers import BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, default_collate
from datasets import Dataset, load_from_disk, concatenate_datasets
#from transformers import BertTokenizer, BertModel, BertForMaskedLM
#import torch
import json

import torch
from torch import optim

from utils import load_vocab, load_data, batchify, evaluate, get_relation_meta

import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

random_seed = random.seed(42)

# setting up the device for GPU usage
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# setting torch default data type
torch.set_default_dtype(torch.float32)

first = True
# Weight and biases log in
wandb.login()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'train_loss'},
    'parameters':
        {
            'batch_size': {'values': [50, 100]},
            'epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 0.01}
        }
}

# Initialize sweep by passing in config.
# (Optional) Provide a name of the project.
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='lm-kbc'
)

MAX_NUM_VECTORS = 20
METRIC_CUT = 300
NEGATIVES = 6
RELATION = ''
def get_new_token(vid):
    assert(vid > 0 and vid <= MAX_NUM_VECTORS)
    return '[V%d]'%(vid)

def convert_manual_to_dense(manual_template, model, tokenizer):
    def assign_embedding(new_token, token):
        """
        assign the embedding of token to new_token
        """
        logger.info('Tie embeddings of tokens: (%s, %s)'%(new_token, token))
        id_a = tokenizer.convert_tokens_to_ids([new_token])[0]
        id_b = tokenizer.convert_tokens_to_ids([token])[0]
        with torch.no_grad():
            model.roberta.embeddings.word_embeddings.weight[id_a] = model.roberta.embeddings.word_embeddings.weight[id_b].detach().clone()
            model.roberta.embeddings.word_embeddings.weight[id_a].require_grad = True

    new_token_id = 0
    template = []
    for word in manual_template.split(" "):
        if word in ['{subject_entity}', '{mask_token}']:
            template.append(word)
        else:
            tokens = tokenizer.tokenize(' ' + word)
            for token in tokens:
                new_token_id += 1
                template.append(get_new_token(new_token_id))
                assign_embedding(get_new_token(new_token_id), token)

    return ' '.join(template)

def init_template(template, model, tokenizer):
    template = convert_manual_to_dense(template, model, tokenizer)
    return template


# Read prompt templates from a CSV file
def read_prompt_templates_from_csv(file_path: str):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
    return prompt_templates


def prepare_for_dense_prompt(bert, tokenizer):
    new_tokens = [get_new_token(i+1) for i in range(MAX_NUM_VECTORS)]
    tokenizer.add_tokens(new_tokens)
    ebd = bert.resize_token_embeddings(len(tokenizer))
    logger.info('# vocab after adding new tokens: %d'%len(tokenizer))


def create_prompt(subject_entity: str, relation: str, prompt_template: dict, tokenizer) -> str:
    prompt = prompt_template[relation].format(subject_entity=subject_entity, mask_token=tokenizer.mask_token)
    return prompt


def create_faiss_two_index(filepath1, filepath2):
    words_dataset = load_from_disk(filepath1)
    entities_dataset = load_from_disk(filepath2)

    dataset = concatenate_datasets([words_dataset, entities_dataset])

    dataset.add_faiss_index(column='embeddings')

    return dataset

def create_faiss_index(filepath1):
    #words_dataset = load_from_disk(filepath1)
    dataset = load_from_disk(filepath1)

    #dataset = concatenate_datasets([words_dataset, entities_dataset])

    dataset.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_L2)

    return dataset

def generate_negatives(X, labels, subjects,  decoder, decoder_averages, subjects_name, index):
    #print(decoder_averages[0].shape)
    """_, retrieved_batch = index.get_nearest_examples_batch('embeddings', decoder_averages, k=NEGATIVES)

    embeddings = []
    label_one_hot = []
    for d, l, r in zip(decoder, labels, retrieved_batch):
        valid_answers = len(d)
        #if not valid_answers:
        #    print(l)
        retrieved_negatives = torch.tensor(r['embeddings'][:-valid_answers])
        d = torch.stack([torch.tensor(v) for v in d], dim=0)
        decod = torch.cat((retrieved_negatives, d), dim=0)
        embeddings.append(decod)
        label_correct_order = r['label'][:-valid_answers] + l
        label_one_hot.append(torch.tensor([1 if r in l else 0 for r in label_correct_order]))
        #print(label_correct_order)
        assert set(l) == set([r for r in label_correct_order if r in l])
        #print(label_one_hot[-1][:5])


    embeddings = torch.stack(embeddings, dim=0)"""

    #labels = torch.stack(label_one_hot, dim=0)

    return X, labels, subjects, decoder, subjects_name, index


def get_batch_examples(data, index=None):
    # batch = default_collate(data)
    input_ids, token_type_ids, attention_mask, subjects, prompt_rep, labels, decoder, decoder_averages, subjects_name = [], [], [], [], [], [], [], [], []
    for d in data:
        input_id = d[0]
        prompt_rep.append(d[3])
        input_ids.append(input_id)
        attention_mask.append(d[1])
        subjects.append(torch.tensor(d[2]))
        labels.append(d[4])
        decoder.append(d[5])
        subjects_name.append(d[6])
        decoder_averages.append(torch.tensor(np.mean(d[5], axis=0)))

    #for i, j in enumerate(prompt_rep):
    #    if j == -1:
    #        continue
    #    input_ids[i][j] = i + 1

    input_ids = torch.stack(input_ids, dim=0)
    #print(input_ids)
    #token_type_ids = torch.stack(token_type_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    X = {'input_ids': input_ids, 'attention_mask': attention_mask}
    #labels = torch.stack(labels, dim=0)
    decoder_averages = torch.stack(decoder_averages, dim=0)
    subjects = torch.stack(subjects, dim=0)
    prompt_rep = torch.tensor(prompt_rep)
    #print(subjects.size())

    return generate_negatives(X, labels, subjects, decoder, decoder_averages.detach().numpy(), subjects_name, index)


def configure_collate_fn(index):
    collate_fn = partial(get_batch_examples, index=index)
    return collate_fn


def load_training_dataset(filepath):
    mapping_train_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            mapping_train_data.append(data)
    return mapping_train_data


def generate_splits(mapping_train_data):
    mapping_train_data_df = pd.DataFrame(mapping_train_data)

    X_df = mapping_train_data_df['bert']
    X = np.array([emb for emb in X_df])

    y_df = mapping_train_data_df['wikipedia2vec']
    y = np.array([emb for emb in y_df])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    return X_train, X_test, y_train, y_test


def generate_dataloader(prompts, subjects, prompt_rep, y, weights, subjects_wiki, batch_size, collate_fn, workers):
    input_ids = torch.tensor(prompts['input_ids'])
    #token_type_ids = torch.tensor(prompts['token_type_ids'])
    attention_mask = torch.tensor(prompts['attention_mask'])  # .to(device).float()
    #y_tensor = torch.tensor(y)  # .to(device).float()
    # weights = torch.tensor(weights)

    dataset = list(zip(input_ids, attention_mask, subjects, prompt_rep, y, weights, subjects_wiki))

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=workers)

    return dataloader


def define_model(lr):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    roberta = RobertaForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True).to(device)


    original_vocab_size = len(list(tokenizer.get_vocab()))
    logger.info('Original vocab size: %d'%original_vocab_size)
    prepare_for_dense_prompt(roberta, tokenizer)

    roberta.train()
    bert = roberta.roberta
    classifier = roberta.lm_head

    linear_mapping = torch.load("./trained_mapping/best_linear_mapping_prompts_normalized_entities_cdist_11.pt").to(device)# .nn.Linear(768, 500, bias=False).to(device) #
    reverse_linear_mapping = torch.load("./trained_mapping/best_linear_mapping_wiki2vec2bert_1.pt").to(device)# .nn.Linear(768, 500, bias=False).to(device) #

    transform_wiki = torch.load(f'trained_mapping/cdist_roberta_transform_wiki.pt').to(device)#.cpu()
    decoder_lm2wiki = torch.load(f'trained_mapping/cdist_roberta_decoder_lm2wiki.pt').to(device)#.cpu()
    transform_lm = torch.load(f'trained_mapping/cdist_roberta_transform_lm.pt').to(device)#.cpu()
    decoder_wiki2lm = torch.load(f'trained_mapping/cdist_roberta_decoder_wiki2lm.pt').to(device)#.cpu()

    model = (transform_wiki, decoder_lm2wiki, transform_lm, decoder_wiki2lm)
    #linear_mapping = torch.nn.Linear(768, 500, bias=True)
    #with torch.no_grad():
    #    linear_mapping.weight = torch.nn.Parameter(pretrained_mapping.weight)
    #linear_mapping = linear_mapping.to(device)
    loss = nn.BCEWithLogitsLoss(reduction='none')  # weight=torch.tensor([1000/2**i for i in range(1000)]))

    lossMSE = nn.MSELoss()

    wiki2vec_retriever = nn.Sequential(
        nn.Linear(500, 500, bias=False),
        nn.Dropout(),
        nn.GELU(),
        nn.LayerNorm(500),
        nn.Linear(500, 500, bias=False),
    ).to(device)

    #with torch.no_grad():
    #    wiki2vec_retriever[1].weight = torch.nn.Parameter(linear_mapping.weight.detach().clone())

    #lr = 0.01


    params =   [] #list(reverse_linear_mapping.parameters())+ list(bert.embeddings.word_embeddings.parameters()) + list(linear_mapping.parameters())# + list(wiki2vec_retriever.parameters())

    for m in model:
        params += list(m.parameters())

    trainer = torch.optim.AdamW(params, lr=lr)#, weight_decay=0.15)

    return roberta, bert, tokenizer, classifier, model, wiki2vec_retriever, reverse_linear_mapping, loss, lossMSE, trainer, original_vocab_size


"""def train_one_step(X, y, decoder, bert, tokenizer, classifier, linear_mapping, wiki2vec_classifier, loss, lossMSE):

    mask_indices = ((X['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]).tolist()

    embd = bert(**X).last_hidden_state
    embd_batch = torch.stack([embd[batch, index, :] for batch, index in enumerate(mask_indices)])

    embd_batch = classifier.predictions.transform(embd_batch)

    wiki2vec = linear_mapping(embd_batch)

    # print(wiki2vec.shape)
    wiki2dec = wiki2vec_classifier(wiki2vec)

    input_logits_batch = torch.einsum('be, bse -> bs', wiki2dec, decoder)
    # sigmoid = nn.Sigmoid()
    # input_probs_batch = sigmoid(input_logits_batch)

    # print(input_logits_batch.shape)
    # print(y)
    weights = []
    y_c = y.clone()
    sums = torch.sum(y_c, dim=1)
    for k, i in enumerate(sums.tolist()):
        if i == 0:
            i += 1
            sums[k] += 1
        weights.append(torch.tensor([1 if j < i * 2 else 0 for j in range(NEGATIVES)]))

    print(decoder.size())
    print(y.size())
    average_correct_embedding = torch.mean(decoder*y.unsqueeze(dim=2), dim=1)
    print(average_correct_embedding.size())

    weights = torch.stack(weights, dim=0).to(device)
    # print(weights.shape)
    # print(y.shape)
    # print(input_logits_batch.shape)
    step_loss = torch.mean(torch.sum(loss(input_logits_batch, y) * weights, dim=1) / (sums * 2))

    step_loss += lossMSE(wiki2vec, average_correct_embedding)
    return step_loss"""

def train_one_step(X, y, decoder, subjects_name, index, bert, tokenizer, classifier, linear_mapping, wiki2vec_retriever, reverse_mapping, loss, lossMSE, subjects):
    """

    Args:
        X:
        y: topK_passage_embeddings
        decoder: topK_passage_embeddings
        bert:
        tokenizer:
        classifier:
        linear_mapping:
        wiki2vec_classifier:
        loss:
        lossMSE:
        wiki2vec: qembedding

    Returns:

    """
    transform_wiki, decoder_lm2wiki, transform_lm, decoder_wiki2lm = linear_mapping
    mask_indices = ((X['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]).tolist()
    subs_needed = False
    embd = bert.embeddings(**{k:v for k, v in X.items() if not k == 'attention_mask'})
    if subs_needed or True:
        subs_indices = ((X['input_ids'] == tokenizer.unk_token_id).nonzero(as_tuple=True)[1]).tolist()
        subs = decoder_wiki2lm(transform_wiki(subjects.to(device))).view(-1, 768)
        subs = subs/torch.norm(subs, dim=1).view(-1, 1)
        for i, s in enumerate(subs_indices):
            if i == 16:
                continue
            embd[i,s,:] = subs[i]

    embd = bert.encoder(embd).last_hidden_state
    embd_batch = torch.stack([embd[batch, index, :] for batch, index in enumerate(mask_indices)], dim=0)

    embd_batch = classifier.layer_norm(classifier.dense(embd_batch))
    #embd_batch = embd_batch/torch.norm(embd_batch, dim=1).view(-1, 1)
    #print("embd_batch", embd_batch)
    wiki2vec = decoder_lm2wiki(transform_lm(embd_batch))
    #wiki2vec = wiki2vec_retriever(wiki2vec)

    #print(y)
    if subs_needed:
        #print("here")
        softmax = nn.Softmax(dim=1)
        cross_entropy = nn.CrossEntropyLoss()
        embd_batch_subs = torch.stack([embd[batch, index, :] for batch, index in enumerate(subs_indices)], dim=0)
        embd_batch_subs = classifier.layer_norm(classifier.dense(embd_batch_subs))
        #y_bert =
        #classifier.predicitions.decoder(embd_batch_subs)

        embd_batch_subs = embd_batch_subs/torch.norm(embd_batch_subs, dim=1).view(-1, 1)

        #print("wiki2vec", wiki2vec)

        #print("wiki2vec", wiki2vec)

        subject_2wiki2vec = linear_mapping(embd_batch_subs)
        #print(subject_2wiki2vec.shape)
        #print(subjects.shape)
        #_, retrieved_batch_subs = index.get_nearest_examples_batch('embeddings', subject_2wiki2vec.detach().cpu().numpy(), k=NEGATIVES)
        _, retrieved_batch_subs = index.get_nearest_examples_batch('embeddings', subjects.view(-1,500).detach().cpu().numpy(), k=NEGATIVES)

        y_subs = [torch.tensor([1 if x == subjects_name[i] else 0 for x in k['label']]) for i, k in enumerate(retrieved_batch_subs)]

        targets = []
        for i, r in enumerate(retrieved_batch_subs):
            if torch.sum(y_subs[i]) == 0:
                targets.append(torch.cat((torch.tensor(r['embeddings'][1:]),  subjects[i].view(1,-1))))
                y_subs[i][-1] = 1
            else:
                targets.append(torch.tensor(r['embeddings']))

        targets = torch.stack(targets, dim=0).to(device)
        y_subs = torch.stack(y_subs, dim=0).to(device).float()
        #print(targets.shape)
        #targets = torch.cat((subjects.unsqueeze(dim=1).to(device), targets), dim=1)
        #print(targets.shape)
        input_logits_batch = torch.cdist(subject_2wiki2vec.unsqueeze(dim=1), targets, p=2).squeeze()
        #print(input_logits_batch.shape)
        min_ = torch.min(input_logits_batch, dim=1)[0].view(-1, 1)

        # print(input_logits_batch[0,:])

        input_logits_batch = torch.exp(-(input_logits_batch-min_))
        # print(input_logits_batch[0,:])
        # print(input_logits_batch)
        #probs = torch.einsum('be, bse -> bs', embd, targets)*100
        #probs = torch.cdist(targets[:, :1, :], targets, p=2).squeeze()
        #max_ = torch.max(probs, dim=1)[0].view(-1, 1)
        # print(max)
        # print(probs.shape)
        #probs = torch.exp(-probs)*100
        # print(probs)
        # print('Input_logits', input_logits_batch[0,:])
        # probs = torch.flip(torch.arange(1, input_logits_batch.size()[1] +1, 1), dims=[0]).float().to(device)
        #probs = softmax(probs)
        # target_prob_batch = probs.unsqueeze(dim=0).expand(input_logits_batch.size()[0], -1)
        # print('Target_prob', target_prob_batch[0,:])
        #print(probs)
        loss_classification = cross_entropy(input_logits_batch, y_subs)

    _, retrieved_batch = index.get_nearest_examples_batch('embeddings', wiki2vec.detach().cpu().numpy(), k=NEGATIVES)
    embeddings = []
    label_one_hot = []
    weights = []
    recalls = []
    for ind, (d, l, r) in enumerate(zip(decoder, y, retrieved_batch)):
        assert len(d) == len(l), print(f"l {l}")
        valid_answers = len(l)
        #if not valid_answers:
        #    print(l)
        retrieved_negatives = torch.tensor(r['embeddings'])
        label_correct_order = r['label']
        needed_labels = [x for x in l if x not in label_correct_order]
        for x in [x for x in label_correct_order if x in l][::-1]:
            if x not in label_correct_order[:-(len(needed_labels)+1)]:
                needed_labels.append(x)
        recalls.append(1-len(needed_labels)/valid_answers)
        already_in_labels = [i for i, x in enumerate(l) if x not in needed_labels]
        if len(needed_labels) > 0:
            d_to_append = [x for i, x in enumerate(d) if i not in already_in_labels]
            l_to_append = [x for i, x in enumerate(l) if i not in already_in_labels]
            #with torch.no_grad():
            dex = torch.stack(list(map(torch.tensor, d_to_append)), dim=0).to(device)
            order = torch.argsort(torch.cdist(wiki2vec.detach().clone()[ind:ind+1, :], dex, p=2)).tolist()[0]
            assert len(needed_labels) == len(l_to_append)
            #print(order)
            #print(needed_labels)

            needed_labels = [l_to_append[i] for i in order]
            assert len(already_in_labels) + len(needed_labels) == valid_answers, print(r['label'], l)
            d = torch.stack([torch.tensor(d_to_append[i]) for i in order], dim=0)
            decod = torch.cat((retrieved_negatives[:-len(needed_labels)], d), dim=0)
            label_correct_order = label_correct_order[:-len(needed_labels)] + needed_labels
            label_one_hot.append(torch.tensor([1 if r in l else 0 for r in label_correct_order]))
            print(r['label'], label_correct_order, l, needed_labels)
            assert torch.sum(label_one_hot[-1]) == valid_answers, print(f"In if len order {label_correct_order} \nlen label {len(label_one_hot[-1])} \ntorch.sum(label_one_hot[-1]) {torch.sum(label_one_hot[-1])} \nalready in {r['label'][:-len(needed_labels)]} \nvalid_answers {valid_answers} \nall_answers {l} \nl_needed_labels {needed_labels}")
        else:
            decod = retrieved_negatives
            label_one_hot.append(torch.tensor([1 if r in l else 0 for r in label_correct_order]))
            assert torch.sum(label_one_hot[-1]) == valid_answers, print(f"In else len order {len(label_correct_order)} len label {len(label_one_hot[-1])} torch.sum(label_one_hot[-1]) {torch.sum(label_one_hot[-1])} valid_answers {valid_answers} needed_labels {needed_labels}")
        assert len(decod) == NEGATIVES, print(f"decod {len(decod)}")
        embeddings.append(decod)
        weights.append(torch.norm(decod, dim=1).view(-1,))
        #print(label_correct_order)
        #assert set(l) == set([r for r in label_correct_order if r in l])
        #print(label_one_hot[-1][:5])

    recall = sum(recalls)/len(recalls)
    decoder = torch.stack(embeddings, dim=0).to(device)

    y = torch.stack(label_one_hot, dim=0).to(device)
    #weights = torch.stack(weights, dim=0).to(device)
    #max_ = torch.max(weights, dim=1)[0].view(-1, 1)
    #weights = (1.01-weights/max_).view(-1, NEGATIVES, 1)
    #print(weights)

    #wiki2vec = wiki2vec/torch.norm(wiki2vec, dim=1).reshape(-1,1)
    #print("t_l.s", y.shape)
    print("y", y[0, :6])
    target_labels = y.view(wiki2vec.size()[0], -1, 1)
    #print("t_l.s", target_labels.shape)
    #print("target_labels", target_labels.shape)
    rel_diff = target_labels - torch.permute(target_labels, (0, 2, 1))
    print("rel_diff", rel_diff.shape)
    pos_pairs = (rel_diff.unsqueeze(2) > 0).float().to(device).view(wiki2vec.size()[0], NEGATIVES, NEGATIVES)
    print("pos_pairs", pos_pairs[0])
    #print("pos_pairs", pos_pairs.shape)
    num_pos_pairs = torch.sum(pos_pairs, dim=2)
    print("pos_pairs", num_pos_pairs)
    #print("pos_pairs", pos_pairs.shape)

    #wiki2dec = wiki2vec_classifier(embd_batch)
    y_pred = torch.cdist(wiki2vec.unsqueeze(dim=1), decoder, p=2).view(wiki2vec.size()[0], -1,1)
    #print("decoder", decoder.requires_grad)
    #print("y_pred", y_pred)
    #print("similarity", y_pred.shape)
    min_ = torch.min(y_pred, dim=1)[0].view(-1, 1, 1).expand(-1, NEGATIVES, -1)
    print("maximum", min_[0][0])
    print("y_pred", y_pred[0, :6, :])
    #print("target_labels", target_labels[0, :6 , :])
    #y_pred = y_pred
    #print("similarity_btw_1", y_pred[0])
    y_pred = torch.exp(-(y_pred-min_))
    #print("similarity", y_pred.shape)
    #print("similarity_1", y_pred[0])
    #y_pred = torch.einsum("ij, ikj -> ik", wiki2vec, decoder).view(wiki2vec.size()[0], -1, 1)
    #print("y_pred", y_pred.shape)

    neg_pairs = (rel_diff.unsqueeze(2) < 0).float().to(device).view(wiki2vec.size()[0], NEGATIVES, NEGATIVES)
    #print("neg_pairs", neg_pairs[0,:6, :])
    num_neg_pairs = torch.sum(neg_pairs, dim=2)
    #print("num_neg_pairs", num_neg_pairs)# num pos pairs and neg pairs are always the same
    #print("num_neg_pairs", num_neg_pairs.shape)
    num_pairs = num_neg_pairs + num_pos_pairs
    #print("num_pairs", num_pairs.shape)

    pos_pairs = pos_pairs.to(device)
    neg_pairs = neg_pairs.to(device)
    sigma = 5 #10/torch.mean(y_pred)
    print("y_pred_after_kernel", y_pred[0, :6])
    y_pred = y_pred - y_pred.permute((0, 2, 1))
    print("y_pred_after_minus", y_pred[0, :6])
    #print("Y_pred_minus", y_pred.shape)
    #logits_max = torch.amax(y_pred, dim=(1, 2))
    #print("logits_max", logits_max.shape)

    #y_pred = y_pred - logits_max.view(-1, 1, 1)
    #print(y_pred[0, :5, :])
    #print("Y_pred", y_pred[0, 0, :])
    #print("Y_pred", target_labels[0, :, :].view(1,-1))
    C_pos = torch.log(1 + torch.exp(-sigma * y_pred))
    #print("c_pos", C_pos[0][0])
    C_neg = torch.log(1 + torch.exp(sigma * y_pred))
    #C_pos = torch.log(1 + torch.exp(-sigma * (y_pred - y_pred.permute((0, 2, 1))))).to(device)
    #C_neg = torch.log(1 + torch.exp(sigma * (y_pred - y_pred.permute((0, 2, 1))))).to(device)
    #print(C_pos.shape)
    #print(C_neg.shape)
    #print(C_pos[0, :5, :])
    #print(C_neg[0, :5, :])

    C = pos_pairs * C_pos + neg_pairs * C_neg
    #print(C[0,:6])

    arr = 1 / (torch.arange(1, 1 + decoder.size()[1]).float().to(device))
    #arr[:3] = 0
    #y[:,:5] = 0
    #y = y.unsqueeze(2).expand(-1 , -1, NEGATIVES)
    #print("y", y.size())
    weights = torch.abs(arr.view(-1, 1) - arr.view(1, -1))
    weights = weights.unsqueeze(0) #.expand(y.shape[0],-1,-1)+y
    #weights[:, METRIC_CUT:] = 0
    #weights = torch.abs(arr.view(-1, 1) - arr.view(1, -1)).view(1, decoder.size()[1], decoder.size()[1]).to(device)
    #weights = (y.unsqueeze(2).expand(-1, -1,  y.shape[1])*arr_big)+arr
    print("weights", weights.size())
    print("C", C.size())
    #C = C * torch.ones_like(pos_pairs) + pos_pairs*4
    C = C*weights
    #C[2:,2:] *= 1.5
    #print("C_sum_weights", torch.sum(C))
    print("num_pairs", num_pairs.shape)
    C = torch.sum(C, dim=2)/num_pairs
    #print("C.sum2", C.shape)
    C = torch.mean(C, dim=1)
    #print("C.sum1", C.shape)
    #print(num_pairs.shape)
    step_loss = torch.mean(C)
    #print(step_loss)
    if subs_needed:
        step_loss += loss_classification
    return step_loss, recall


def train(num_epochs, train_dataloader, test_dataloader, bert, tokenizer, classifier, linear_mapping, wiki2vec_retriever,
          reverse_mapping, loss, lossMSE, trainer, original_vocab_size):
    best_epoch = 0
    best_valid_recall = 0
    best_valid_loss = 100000000
    # number of batches in train and test split
    n_train_batches = len(train_dataloader)
    n_test_batches = len(test_dataloader)

    training_range = tqdm(range(num_epochs), desc='Epoch', position=0)
    step_range = tqdm(range(n_train_batches), desc='Batch', position=0)

    for epoch in range(num_epochs):

        total_epoch_loss = 0
        total_valid_recall = 0
        total_valid_loss = 0

        for step, data in enumerate(train_dataloader):
            X, y, subjects, decoder, subjects_name, index = data
            # print(X.size())
            # print(y.size())

            X = {k: v.to(device) for k, v in X.items()}

            wiki2vec_retriever.train()
            trainer.zero_grad()  # sets gradients to zero

            step_loss,_ = train_one_step(X, y, decoder, subjects_name, index,
                                       bert, tokenizer, classifier, linear_mapping, wiki2vec_retriever, reverse_mapping, loss, lossMSE, subjects)

            total_epoch_loss += step_loss


            step_loss.backward()  # back propagation

            for p in bert.embeddings.word_embeddings.parameters():
                # only update new tokens
                p.grad[:original_vocab_size] = 0.0
                break



            try:
                for p in linear_mapping.parameters():
                # only update new tokens
                    print("linear_mapping", torch.mean(p.grad))
            except:
                pass
                #for p in wiki2vec_retriever.parameters():
                    # only update new tokens
                #    print("wiki", torch.mean(p.grad))
            try:
                for p in reverse_mapping.parameters():
                # only update new tokens
                    print("reverse_mapping", torch.mean(p.grad))
            except:
                pass

            #torch.nn.utils.clip_grad_norm_(bert.parameters(), 1)
            #torch.nn.utils.clip_grad_norm_(wiki2vec_retriever.parameters(), 1)
            #torch.nn.utils.clip_grad_norm_(linear_mapping.parameters(), 1)
            #torch.nn.utils.clip_grad_norm_(reverse_mapping.parameters(), 1)

            trainer.step()  # parameter update


            step_range.set_description(
                "Epoch %d | Step %d | Step loss: %f"
                % (epoch + 1, step + 1, step_loss)
            )
            step_range.update(1)
            wandb.log({
                'loss': step_loss,
            })

        epoch_loss = total_epoch_loss / n_train_batches

        for step, data in enumerate(test_dataloader):
            X, y, subjects, decoder, subjects_name, index = data
            X = {k: v.to(device) for k, v in X.items()}
            wiki2vec_retriever.eval()

            with torch.no_grad():
                step_loss, step_recall = train_one_step(X, y, decoder, subjects_name, index,
                                           bert, tokenizer, classifier, linear_mapping, wiki2vec_retriever, reverse_mapping, loss, lossMSE, subjects)

            total_valid_loss += step_loss
            total_valid_recall += step_recall

        valid_loss = total_valid_loss / n_test_batches
        valid_recall = total_valid_recall / n_test_batches
        relation = 'CompoundHasParts'
        """if valid_recall > best_valid_recall:
            best_valid_recall = valid_recall
            best_epoch = epoch
            print(f"saving {best_epoch}")
            transform_wiki, decoder_lm2wiki, transform_lm, decoder_wiki2lm = linear_mapping
            torch.save(transform_wiki, f'finetuned_mapping/roberta_transform_wiki_{RELATION}_opti.pt')
            torch.save(decoder_lm2wiki, f'finetuned_mapping/roberta_decoder_lm2wiki_{RELATION}_opti.pt')
            torch.save(transform_lm, f'finetuned_mapping/roberta_transform_lm_{RELATION}_opti.pt')
            torch.save(decoder_wiki2lm, f'finetuned_mapping/roberta_decoder_wiki2lm_{RELATION}_opti.pt')
            vs = bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
            with open(os.path.join("finetuned_mapping", f'prompt_vecs_all_roberta_{RELATION}_opti.npy'), 'wb') as f:
                np.save(f, vs)"""
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            print(f"saving {best_epoch}")
            transform_wiki, decoder_lm2wiki, transform_lm, decoder_wiki2lm = linear_mapping
            torch.save(transform_wiki, f'finetuned_mapping/roberta_transform_wiki_{RELATION}_opti.pt')
            torch.save(decoder_lm2wiki, f'finetuned_mapping/roberta_decoder_lm2wiki_{RELATION}_opti.pt')
            torch.save(transform_lm, f'finetuned_mapping/roberta_transform_lm_{RELATION}_opti.pt')
            torch.save(decoder_wiki2lm, f'finetuned_mapping/roberta_decoder_wiki2lm_{RELATION}_opti.pt')
            vs = bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
            with open(os.path.join("finetuned_mapping", f'prompt_vecs_all_roberta_{RELATION}_opti.npy'), 'wb') as f:
                np.save(f, vs)


        training_range.set_description(
            "Epoch %d | Epoch Loss: %f | Validation Loss: %f | Validation Recall: %f"
            % (epoch + 1, epoch_loss, valid_loss, valid_recall)
        )

        training_range.update(1)

        wandb.log({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'valid_loss': valid_loss
        })
        with open("./best_model.json", "w") as f:
            json.dump({'CityLocatedRiver': best_epoch}, f)



def get_vector(x, replace):
    if x == 'Random':
        return list([0.0 for _ in range(500)])
    return replace[x]


def main():
    global NEGATIVES, RELATION
    run = wandb.init()

    '''#hyperparameters
    logger.info(f"Setting hyperparameters...")
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    num_epochs = wandb.config.epochs'''

    # hyperparameters
    logger.info(f"Setting hyperparameters...")
    batch_size = 16
    lr = 0.001
    num_epochs = 20

    # load dataset
    logger.info(f"Loading dataset...")
    mapping_train_data = load_training_dataset('./data/train.jsonl')
    mapping_val_data = load_training_dataset('./data/val.jsonl')

    prompt_templates = read_prompt_templates_from_csv("prompts_meta.csv")

    logger.info(f"Loaded \"{len(mapping_train_data)}\" rows...")
    # defining the model, loss function and Trainer
    logger.info(f"Loading model, loss and trainer...")
    relations = {'CountryHasOfficialLanguage': 20,
                 'CountryBordersCountry': 20,
                 'CompoundHasParts': 10,
                 'BandHasMember': 20,
                 'CityLocatedAtRiver': 6,
                 'CompanyHasParentOrganisation': 10,
                 'FootballerPlaysPosition': 6,
                 'CountryHasStates': 50,
                 'PersonCauseOfDeath': 6,
                 'PersonHasAutobiography': 6,
                 'PersonHasEmployer': 10,
                 'PersonHasNoblePrize': 6,
                 'PersonHasPlaceOfDeath': 6,
                 'PersonHasProfession': 10,
                 'PersonHasSpouse': 10,
                 'PersonPlaysInstrument': 20,
                 'PersonSpeaksLanguage': 20,
                 'RiverBasinsCountry': 10,
                 'StateBordersState': 20,
    }

    for r, n in relations.items():
        model, bert, tokenizer, classifier, linear_mapping, wiki2vec_retriever, reverse_mapping, loss, lossMSE, trainer, original_vocab_size = define_model(
            lr)
        NEGATIVES = n
        RELATION = r
        relations_to_train = [RELATION]
        template = init_template(prompt_templates[relations_to_train[0]], model, tokenizer)
        opti_prompt_templates = {relations_to_train[0]: template}
        # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        with open("./data/wikidata_2_wiki2vec.json") as f:
            wikidata2wiki2vec = json.load(f)

        #e_data = datasets.load_from_disk('/data_ssds/disk08/biswasdi/lm-kbc-data/entities_faiss')
        #replacement_entities = dict(zip(e_data['label'], e_data['embeddings']))
        # loading the wikipedia2vec dictionary
        replacement_entities = {}
        with open("./data/replacement_entities_new.json") as f:
            replacement_entities = json.load(f)

        """
        with open("/data_ssds/disk08/biswasdi/lm-kbc-data/enwiki_20180420_500d.txt", 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                token = values[0]
                vector = np.asarray(values[-500:], "float32")
                # embeddings_dict[token] = vector
    
                if "ENTITY/" in token:
                    replacement_entities[token] = vector
        """

        wikidata2wiki2vec[''] = 'Random'

        # replacement_entities = {v: torch.rand((500,)) for v in wikidata2wiki2vec.values()}

        reduced_train_data = []
        for m in mapping_train_data:
            if not m['Relation'] in relations_to_train:
                continue
            if any([wikidata2wiki2vec.get(o) for o in m['ObjectEntitiesID']]) and wikidata2wiki2vec.get(m['SubjectEntityID']):
                m['ObjectWikiEntities'] = [wikidata2wiki2vec.get(o) for o in m['ObjectEntitiesID'] if wikidata2wiki2vec.get(o)]
                m['prompt'] = create_prompt(m['SubjectEntity'], m['Relation'], opti_prompt_templates, tokenizer) #" ".join([, '([UNK])'])
                m['EntityVector'] = [get_vector(o, replacement_entities) for o in m['ObjectWikiEntities'] if o]
                m['SubjectWikiEntities'] = [wikidata2wiki2vec.get(m['SubjectEntityID'])]
                m['SubjectEntityVector'] = [get_vector(o, replacement_entities) for o in m['SubjectWikiEntities']]
                m['label'] = len([x for x in m['ObjectWikiEntities'] if not x == 'Random'])
                reduced_train_data.append(m)

        reduced_val_data = []
        for m in mapping_val_data:
            if not m['Relation'] in relations_to_train:
                continue
            if any([wikidata2wiki2vec.get(o) for o in m['ObjectEntitiesID']]) and wikidata2wiki2vec.get(m['SubjectEntityID']):
                m['ObjectWikiEntities'] = [wikidata2wiki2vec.get(o) for o in m['ObjectEntitiesID'] if wikidata2wiki2vec.get(o)]
                m['prompt'] = create_prompt(m['SubjectEntity'], m['Relation'], opti_prompt_templates, tokenizer) #m['SubjectEntity']
                m['EntityVector'] = [get_vector(o, replacement_entities) for o in m['ObjectWikiEntities']]
                m['SubjectWikiEntities'] = [wikidata2wiki2vec.get(m['SubjectEntityID'])]
                m['SubjectEntityVector'] = [get_vector(o, replacement_entities) for o in m['SubjectWikiEntities']]
                m['label'] = len([x for x in m['ObjectWikiEntities'] if not x == 'Random'])
                reduced_val_data.append(m)

        #entities_faiss_index = create_faiss_index('/data_ssds/disk08/biswasdi/lm-kbc-data/entities_faiss')

        # generate train test splits
        logger.info(f"Split datasets into train and test...")
        # X_train, X_test, y_train, y_test = generate_splits(mapping_train_data)
        #reverse_mapping = torch.load("trained_mapping/best_linear_mapping_wiki2vec2bert_0.pt").cpu()


        prompts = tokenizer([m['prompt'] for m in reduced_train_data], truncation=True, padding='longest')
        print(prompts['input_ids'])
        labels = [m['ObjectWikiEntities'] for m in reduced_train_data]
        weights = [m['EntityVector'] for m in reduced_train_data]
        subjects = [m['SubjectEntityVector'] for m in reduced_train_data]
        subjects_wiki = [m['SubjectWikiEntities'] for m in reduced_train_data]
        prompts_rep = [p.index(100) if 100 in p else -1 for p in prompts['input_ids']]
        # create faiss index
        logger.info(f"Loaded Wikipedia2vec datasets...")

        dataset_faiss_index = create_faiss_index('/data_ssds/disk08/biswasdi/lm-kbc-data/entities_faiss')
        collate_fn = configure_collate_fn(dataset_faiss_index)

        # load dataloader
        logger.info(f"Load dataloaders for train and test...")

        train_dataloader = generate_dataloader(prompts, subjects, prompts_rep, labels, weights, subjects_wiki, batch_size, collate_fn, 1)

        prompts_val = tokenizer([m['prompt'] for m in reduced_val_data], truncation=True, padding='longest')
        labels_val = [m['ObjectWikiEntities'] for m in reduced_val_data]
        weights_val = [m['EntityVector'] for m in reduced_val_data]
        subjects_val = [m['SubjectEntityVector'] for m in reduced_val_data]
        subjects_wiki_val = [m['SubjectWikiEntities'] for m in reduced_val_data]
        prompts_rep_val = [p.index(100) if 100 in p else 0 for p in prompts['input_ids']]
        test_dataloader = generate_dataloader(prompts_val, subjects_val, prompts_rep_val, labels_val, weights_val, subjects_wiki_val, batch_size, collate_fn, 1)

        # start training
        logger.info(f"Start training...")
        train(num_epochs, train_dataloader, test_dataloader, bert, tokenizer, classifier, linear_mapping, wiki2vec_retriever, reverse_mapping, loss, lossMSE, trainer, original_vocab_size)

if __name__ == "__main__":
    # wandb.agent(sweep_id, function=main, count=1)

    main()

