# importing necessary libaries
import json

import faiss

import wandb
import torch
import random
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from transformers import RobertaTokenizer, RobertaForMaskedLM
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, default_collate
from datasets import Dataset, load_from_disk, concatenate_datasets


# setting random seed
random_seed = random.seed(42)

# setting up the device for GPU usage
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# setting torch default data type
torch.set_default_dtype(torch.float32)

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

"""
class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        #self.ent_index = entities_faiss_index
        #self.wp_index = wordpiece_faiss_index
        self.
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        
        #print(inputs.size())
        #print(targets.size())
        
        target_prob_batch = []
        input_logits_batch = []
        retrieved_examples_batch = []
        targets_numpy = np.array([t.detach().cpu().numpy() for t in targets])

        entities_scores, retrieved_entities_batch = self.ent_index.get_nearest_examples_batch('embeddings', targets_numpy, k=500)
        words_scores, retrieved_words_batch = self.wp_index.get_nearest_examples_batch('embeddings', targets_numpy, k=500)

        for pred, target, retrieved_entities, retrieved_words in zip(inputs, targets, retrieved_entities_batch, retrieved_words_batch):

            #print(target_list.shape)

            retrieved_examples = {'Entity': retrieved_entities['Entity'] + retrieved_words['Entity'],
                                 'label': retrieved_entities['label'] + retrieved_words['label'],
                                 'embeddings': retrieved_entities['embeddings'] + retrieved_words['embeddings']}

            dot_products_bert = []
            dot_products_wiki = []

            for example in retrieved_examples['embeddings']:

                example = torch.tensor(example).to(device)

                dp_bert = torch.dot(pred, example)

                dot_products_bert.append(dp_bert)

                dp_wiki = torch.dot(target, example)

                dot_products_wiki.append(dp_wiki)

            retrieved_examples['dot_products_bert'] = dot_products_bert
            retrieved_examples['dot_products_wiki'] = dot_products_wiki

            retrieved_examples_batch.append(retrieved_examples)

            target_prob = self.softmax(torch.stack(dot_products_wiki, dim = 0).to(device))
            #print(target_prop.shape)

            input_logits = torch.stack(dot_products_bert, dim = 0).to(device)

            target_prob_batch.append(target_prob)
            input_logits_batch.append(input_logits)
        #inputs = inputs/torch.norm(inputs, dim=1).view(-1, 1)

        #input_logits_batch = torch.einsum('be, bse -> bs', inputs, targets)
        #print(inputs)
        input_logits_batch = torch.cdist(inputs.unsqueeze(dim=1), targets, p=2).squeeze()

        #print(input_logits_batch[0,:])
        min_ = torch.max(input_logits_batch, dim=1)[0].view(-1, 1)
        #print(max_)
        input_logits_batch = torch.exp(-(input_logits_batch-min_))
        #print(input_logits_batch[0,:])
        #print(input_logits_batch)
        #probs = torch.einsum('be, bse -> bs', targets[:,0,:], targets)*100
        probs = torch.cdist(targets[:,:1,:], targets, p=2).squeeze()
        min_ = torch.min(probs, dim=1)[0].view(-1, 1)
        #print(max)
        #print(probs.shape)
        probs = torch.exp(-(probs-min_)*100
        #print(probs)
        #print('Input_logits', input_logits_batch[0,:])
        #probs = torch.flip(torch.arange(1, input_logits_batch.size()[1] +1, 1), dims=[0]).float().to(device)
        target_prob_batch = self.softmax(probs)
        #target_prob_batch = probs.unsqueeze(dim=0).expand(input_logits_batch.size()[0], -1)
        #print('Target_prob', target_prob_batch[0,:])
                                         
        loss = self.cross_entropy(input_logits_batch, target_prob_batch)

        input_logits_batch = torch.einsum('be, bse -> bs', targets[:,0,:], bert_)
        # print(inputs)
        # input_logits_batch = torch.dot(inputs.unsqueeze(dim=1), targets).squeeze()

        # print(input_logits_batch[0,:])
        # max_ = torch.max(input_logits_batch, dim=1)[0].view(-1, 1)
        # print(max_)
        # input_logits_batch = torch.exp(input_logits_batch)
        # print(input_logits_batch[0,:])
        # print(input_logits_batch)
        probs = torch.einsum('be, bse -> bs', targets[:, 0, :], targets) * 100
        # probs = torch.dot(targets[:,:1,:], targets).squeeze()
        # max_ = torch.max(probs, dim=1)[0].view(-1, 1)
        # print(max)
        # print(probs.shape)
        # probs = probs**2
        # print(probs)
        # print('Input_logits', input_logits_batch[0,:])
        # probs = torch.flip(torch.arange(1, input_logits_batch.size()[1] +1, 1), dims=[0]).float().to(device)
        target_prob_batch = self.softmax(probs)
        # target_prob_batch = probs.unsqueeze(dim=0).expand(input_logits_batch.size()[0], -1)
        # print('Target_prob', target_prob_batch[0,:])

        loss_other = self.cross_entropy(input_logits_batch, target_prob_batch)

        loss += loss_other
        return loss"""


def create_faiss_index(filepath1, filepath2):
    
    words_dataset = load_from_disk(filepath1)
    entities_dataset = load_from_disk(filepath2)
    
    dataset = concatenate_datasets([words_dataset, entities_dataset])
    
    dataset.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_L2)
    
    return dataset  

def generate_negatives(X, y, index):
    
    _, retrieved_batch = index.get_nearest_examples_batch('embeddings', y.detach().numpy(), k=300)

    embeddings = torch.stack([torch.tensor(r['embeddings']) for r in retrieved_batch], dim=0)
    #embeddings = torch.cat((torch.tensor(y).unsqueeze(dim=1), embeddings), dim=1)
    #print(embeddings.size())
    return X.float(), y.float(), embeddings.float()

def get_batch_examples(data, index=None):
    #batch = default_collate(data)
    X, y = [], []
    for d in data:
        X.append(d[0])
        y.append(d[1])

    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)
    #print(X.size())
    #print(y.shape)
    return generate_negatives(X, y, index)
    
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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, random_state=random_seed)
    
    return X_train, X_test, y_train, y_test

def generate_dataloader(X, y, batch_size, collate_fn, workers):
    
    X_tensor = torch.tensor(X) #.to(device).float()
    y_tensor = torch.tensor(y) #.to(device).float()
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=workers)
    
    return dataloader

def define_model(lr):
    
    decoder_wiki2lm = nn.Linear(500, 768, bias = False).to(device)
    decoder_lm2wiki = nn.Linear(768, 500, bias = False).to(device)

    decoder_wiki2lm.weight = torch.nn.Parameter(decoder_lm2wiki.weight.T)

    transform_wiki = nn.Sequential(
        nn.Linear(500, 500),
        nn.LayerNorm(500),
    ).to(device)

    transform_lm = nn.Sequential(
        nn.Linear(768, 768),
        nn.LayerNorm(768),
    ).to(device)

    #model.weight.data.normal_(0, 0.01)
    
    #model.bias.data.fill_(0)
    
    # mean squared error loss function
    loss = None
    params = list(decoder_lm2wiki.parameters()) + list(transform_wiki.parameters()) + list(transform_lm.parameters())
    # implements a stochastic gradient descent optimization method
    trainer = torch.optim.AdamW(params, lr=lr)#, weight_decay=0.01)
    roberta = RobertaForMaskedLM.from_pretrained("roberta-base", output_hidden_states=True)
    roberta_embeddings = roberta.roberta.embeddings.word_embeddings.weight.detach()#.to(device)
    model = (transform_wiki, decoder_lm2wiki, transform_lm, decoder_wiki2lm, roberta_embeddings)
    return model, loss, trainer

def calc_distance(_inputs, _targets, _transform, _decoder,  _target=None):
    _inputs = _transform(_inputs)
    _inputs = _decoder(_inputs)


    if len(_targets.shape) == 3:
        input_logits_batch = torch.cdist(_inputs.unsqueeze(dim=1), _targets.to(device), p=2).squeeze()
    else:
        input_logits_batch = torch.cdist(_inputs, _targets.to(device), p=2).squeeze()

    min_ = torch.max(input_logits_batch, dim=1)[0].view(-1, 1)

    input_logits_batch = torch.exp(-(input_logits_batch - min_))

    if _target == None:
        probs = torch.cdist(_targets[:, :1, :], _targets, p=2).squeeze()
    else:
        probs = []
        for t in _target:
            dis = torch.cdist(t.view(1, 1, -1), _targets.unsqueeze(0).to(device), p=2).squeeze()
            #print(dis.shape)
            probs.append(dis)
        _targets.cpu()
        probs = torch.stack(probs, dim=0)

    min_ = torch.min(probs, dim=1)[0].view(-1, 1)

    probs = torch.exp(-(probs - min_))*100

    target_prob_batch = torch.nn.Softmax(1)(probs)

    loss = torch.nn.CrossEntropyLoss()(input_logits_batch, target_prob_batch)
    _targets.cpu()
    return loss

def calc_loss(model, X, Y, wiki_decoder):

    transform_wiki, decoder_lm2wiki, transform_lm, decoder_wiki2lm, roberta_embeddings = model
    wiki2lm = calc_distance(Y, roberta_embeddings, transform_wiki, decoder_wiki2lm, X)
    lm2wiki = calc_distance(X, wiki_decoder, transform_lm, decoder_lm2wiki)
    return wiki2lm + lm2wiki

def train(num_epochs, train_dataloader, test_dataloader, model, loss, trainer):
    best_epoch = 0
    best_valid_loss = 10000000000
    
    #number of batches in train and test split
    n_train_batches = len(train_dataloader)
    n_test_batches = len(test_dataloader)
    
    training_range = tqdm(range(num_epochs), desc = 'Epoch', position = 0)
    step_range = tqdm(range(n_train_batches), desc = 'Batch', position = 0)

    for epoch in range(num_epochs):

        total_epoch_loss = 0

        total_valid_loss =  0

        for step, data in enumerate(train_dataloader):

            X, Y, wikidec = data
            #print(X.size())
            #print(y.size())
            
            trainer.zero_grad() #sets gradients to zero
            
            step_loss = calc_loss(model,
                                  X.to(device),
                                  Y.to(device),
                                  wikidec.to(device))

            total_epoch_loss += step_loss

            step_loss.backward() # back propagation

            trainer.step() # parameter update

            step_range.set_description(
                        "Epoch %d | Step %d | Step loss: %f"
                        % (epoch + 1, step + 1, step_loss)
                    )
            step_range.update(1)
            wandb.log({
            'loss': step_loss,
              })

        epoch_loss = total_epoch_loss/n_train_batches

        for step, data in enumerate(test_dataloader):
            X, Y, wikidec = data

            with torch.no_grad():
                step_loss = calc_loss(model,
                                      X.to(device),
                                      Y.to(device),
                                      wikidec.to(device))

            total_valid_loss += step_loss

        valid_loss = total_valid_loss/n_test_batches

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            transform_wiki, decoder_lm2wiki, transform_lm, decoder_wiki2lm, roberta_embeddings = model

            torch.save(transform_wiki, f'trained_mapping/cdist_roberta_transform_wiki.pt')
            torch.save(decoder_lm2wiki, f'trained_mapping/cdist_roberta_decoder_lm2wiki.pt')
            torch.save(transform_lm, f'trained_mapping/cdist_roberta_transform_lm.pt')
            torch.save(decoder_wiki2lm, f'trained_mapping/cdist_roberta_decoder_wiki2lm.pt')

        training_range.set_description(
                        "Epoch %d | Epoch Loss: %f | Validation Loss: %f"
                        % (epoch + 1, epoch_loss, valid_loss)
                    )
        
        training_range.update(1)
        
        wandb.log({
        'epoch': epoch,
        'train_loss': epoch_loss,
        'valid_loss': valid_loss
          })
        
        #torch.save(model, f'trained_mapping/test_model_{epoch}.pt')
        
        
def main():
    
    run = wandb.init()
    
    '''#hyperparameters
    logger.info(f"Setting hyperparameters...")
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    num_epochs = wandb.config.epochs'''
    
    #hyperparameters
    logger.info(f"Setting hyperparameters...")
    batch_size = 32
    lr = 0.001
    num_epochs = 20
    
    #load dataset
    logger.info(f"Loading dataset...")
    mapping_train_data = load_training_dataset('data/mapping_train_data_entities_roberta.jsonl')

    logger.info(f"Loaded \"{len(mapping_train_data)}\" rows...")
    
    #create faiss index
    logger.info(f"Loaded Wikipedia2vec datasets...")
    dataset_faiss_index = create_faiss_index('/data_ssds/disk08/biswasdi/lm-kbc-data/entities_faiss',
                                             '/data_ssds/disk08/biswasdi/lm-kbc-data/words_faiss')
    #entities_faiss_index = create_faiss_index('/data_ssds/disk08/biswasdi/lm-kbc-data/entities_faiss')
    
    collate_fn = configure_collate_fn(dataset_faiss_index)
    #generate train test splits
    logger.info(f"Split datasets into train and test...")
    X_train, X_test, y_train, y_test = generate_splits(mapping_train_data)
    
    #load dataloader
    logger.info(f"Load dataloaders for train and test...")
    train_dataloader = generate_dataloader(X_train, y_train, batch_size, collate_fn, 1)
    test_dataloader = generate_dataloader(X_test, y_test, batch_size, collate_fn, 1)
    
    #defining the model, loss function and Trainer
    logger.info(f"Loading model, loss and trainer...")
    model, loss, trainer = define_model(lr)
    
    #start training
    logger.info(f"Start training...")
    train(num_epochs, train_dataloader, test_dataloader, model, loss, trainer)
    
    
    
if __name__ == "__main__":
    #wandb.agent(sweep_id, function=main, count=1)
    
    
    main()

