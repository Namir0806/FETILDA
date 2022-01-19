from scipy import stats
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import os
import random
import sys
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer 
from transformers import AdamW
from torch.cuda.amp import autocast
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import tensorflow as tf

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression

start = time.time()

torch.cuda.empty_cache()

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
tf.random.set_seed(seed_val)


class BERT_Arch(nn.Module):

    def __init__(self, bert):

        super(BERT_Arch, self).__init__()

        self.bert = bert 
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)
      
        # relu activation function
        self.relu =  nn.LeakyReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,600)
      
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(601,1)

        self.fc3 = nn.Linear(1,1)

        #LSTM
        self.hidden_dim = 300
        self.emb_dim = 768
        self.encoder = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, bidirectional=True, dropout=0.1)


    #Define Attention Network
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden
    

    #define the forward pass
    def forward(self, sent_id, mask, hist):

        cls_vec = []
        chunk_max_weights = []
        for i in range(len(sent_id)):

            if i < 35:

                #print("chunk i: ", i)

                ip_id = torch.tensor(sent_id[i]).unsqueeze(0).to(device)
                attn_mask = torch.tensor(mask[i]).unsqueeze(0).to(device)

                #pass the inputs to the model  
                model_outputs = self.bert(input_ids=ip_id, attention_mask=attn_mask)
                cls_hs=model_outputs[1]
                atten=model_outputs[2]
                cls_vec.append(cls_hs)

                '''
                col_sum = np.sort(atten[0][0][11].sum(0)[1:-1].detach().cpu().numpy()) 
                col_sum = col_sum[::-1]  
                max_col_sum = max(col_sum) 
                top_word_mean = col_sum[:15].mean()
                chunk_max_weights.append(top_word_mean)
                '''
        cls_vec_ = torch.max(torch.stack(cls_vec, dim=0), dim=0)   
        
        '''
        cls_vec = torch.stack(cls_vec, dim=0)
        chunk_weights = (torch.tensor(chunk_max_weights)).unsqueeze(0)
        chunk_weights = chunk_weights.cuda()    
        prod1 = torch.bmm(cls_vec.transpose(1,2), chunk_weights.transpose(0,1).unsqueeze(1)) 
        prod1 = prod1.transpose(1,2) 
        
        cls_vec_ = torch.max(prod1, 0) #torch.stack(cls_vec, dim=0), dim=0)
        '''

        #cls_vec = torch.stack(cls_vec, dim=0)
        #cls_vec = cls_vec.to(torch.float32)  #LSTM
        #print("cls_vec shape: ", cls_vec.shape, type(cls_vec), cls_vec.dtype)

        
        x = self.fc1(cls_vec_[0])
        x = self.relu(x)
        x = self.dropout(x)
        
        '''

        emb_input = cls_vec
        inputx = self.dropout(emb_input)
        output, (hn, cn) = self.encoder(inputx) #emb_input)
        fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] #sum bidir outputs F+B
        fbout = fbout.permute(1,0,2)
        fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
        attn_out = self.attnetwork(fbout, fbhn)

        x = attn_out
        '''
        
        hist = hist.unsqueeze(0)

        hist = self.fc3(hist)
        x = torch.cat((x, hist.unsqueeze(0)), dim=1)
        #x = self.dropout(x)

        # output layer
        y = self.fc2(x)

        df_xy=pd.DataFrame()
        df_xy["x"]=[x]
        df_xy["y"]=[y]
        df_xy.to_csv('xy_bert_'+str(max_length)+'_'+sec+'_'+bv+'_max_hist_ep5.csv')

        return x, y


# function to train the model
def train():

    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds = []

    total_hist = []

    xs = []

    # iterate over list of documents
    for i in range(len(train_seq)):

        sent_id = train_seq[i]
        mask = train_mask[i]
        hist = train_hist[i] 
        labels = train_y[i].unsqueeze(0).unsqueeze(0)

        # clear previously calculated gradients 
        model.zero_grad()        

        with autocast():
            # get model predictions for the current batch
            x, preds = model(sent_id, mask, hist)

            # compute the loss between actual and predicted values
            loss = mse_loss(preds, labels)

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()
            x = x.detach().cpu().numpy().ravel()

            # add on to the total loss
            total_loss = total_loss + loss.item()

            xs.append(x)

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # append the model predictions
        total_preds.append(preds)

        
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_seq)

    xs = np.array(xs)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    #total_hist = np.concatenate(total_hist, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds , xs


def test():

    # empty list to save the model predictions
    total_xs = []

    total_preds=[]
    

    for i in range(len(test_seq)):

        sent_id = test_seq[i]
        mask = test_mask[i]
        hist = test_hist[i]
        #labels = test_y[i].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            with autocast():
                x, preds = model(sent_id, mask, hist)
                x = x.detach().cpu().numpy().ravel()
                preds = preds.detach().cpu().numpy()
                # append the model predictions
                total_xs.append(x)
                total_preds.append(preds)

        
            
    # reshape the predictions in form of (number of samples, no. of classes)
    total_xs = np.array(total_xs)

    total_preds = np.concatenate(total_preds, axis=0)
    
    return total_xs, total_preds

# specify GPU
device = torch.device("cuda")

max_length = int(sys.argv[1]) #append two [CLS] and [SEP] tokens to make 512
sec = sys.argv[2]
bv = sys.argv[3]

fname = "sorted_"+ sec + ".csv"

#end_year = int(sys.argv[1])
#train_years_list = list(range(end_year-5, end_year))
#print("train_years: ", train_years_list)

df = pd.read_csv(fname)
#df = df[:10]

train_cik, test_cik, train_text, test_text, train_hist, test_hist, train_labels, test_labels = train_test_split(
    df['cik_year'],
    df['mda'],
    df['prev_'+bv], 
    df[bv],
    shuffle=False,
    test_size=0.2) 

train_text = train_text.astype(str) 
test_text = test_text.astype(str)

'''
df_train = pd.DataFrame()
df_test = pd.DataFrame()

for y in train_years_list:
    df_train = pd.concat([df_train, pd.read_csv(str(y) + "_tok.csv")])
'''

bert_path = "/gpfs/u/home/DLTM/DLTMboxi/scratch/env"

config = AutoConfig.from_pretrained(bert_path + "/bert-base-uncased/", output_attentions=True) 

# import BERT-base pretrained model
bert = AutoModel.from_pretrained(bert_path + "/" + 'bert-base-uncased/', config=config) #longformer-base-4096/') 

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_path + "/" + 'bert-base-uncased/') #longformer-base-4096/') 

#TRAIN
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    add_special_tokens=False
)

#Extract input ids
train_seq_ = tokens_train['input_ids']
#Split each document into 510 tokens
train_seq = [[train_seq_[j][i:i + max_length] for i in range(0, len(train_seq_[j]), max_length)] for j in range(len(train_seq_))]
#print(train_seq[0][0])
#Add [CLS], [SEP] and [PAD] tokens
train_seq = [[[tokenizer.cls_token_id] + train_seq[j][i] + [tokenizer.sep_token_id] if len(train_seq[j][i]) == max_length else [tokenizer.cls_token_id] + train_seq[j][i] +[tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (max_length-len(train_seq[j][i])) for i in range(len(train_seq[j]))] for j in range(len(train_seq))]
#print(train_seq[0][0])
#df_train_seq=pd.DataFrame()
#df_train_seq["train_seq"]=train_seq
#df_train_seq.to_csv(sec+ "-train_seq.csv")

#Extract attention masks
train_mask_ = tokens_train['attention_mask']
#Split each document into 510 tokens
train_mask = [[train_mask_[j][i:i + max_length] for i in range(0, len(train_mask_[j]), max_length)] for j in range(len(train_mask_))]
#Add [1] for attention and [0] for [PAD]
train_mask = [[[1] + train_mask[j][i] + [1] if len(train_mask[j][i]) == max_length else [1]+train_mask[j][i]+[1] + [0] * (max_length-len(train_mask[j][i])) for i in range(len(train_mask[j]))] for j in range(len(train_mask))]


#TEST
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    add_special_tokens=False
)

#Extract input ids
test_seq_ = tokens_test['input_ids']
#Split each document into 510 tokens
test_seq = [[test_seq_[j][i:i + max_length] for i in range(0, len(test_seq_[j]), max_length)] for j in range(len(test_seq_))]
#Add [CLS], [SEP] and [PAD] tokens
test_seq = [[[tokenizer.cls_token_id] + test_seq[j][i] + [tokenizer.sep_token_id] if len(test_seq[j][i]) == max_length else [tokenizer.cls_token_id]+test_seq[j][i] + [tokenizer.sep_token_id]+ [tokenizer.pad_token_id] * (max_length-len(test_seq[j][i])) for i in range(len(test_seq[j]))] for j in range(len(test_seq))]


#Extract attention masks
test_mask_ = tokens_test['attention_mask']
#Split each document into 510 tokens
test_mask = [[test_mask_[j][i:i + max_length] for i in range(0, len(test_mask_[j]), max_length)] for j in range(len(test_mask_))]
#Add [1] for attention and [0] for [PAD]
test_mask = [[[1] + test_mask[j][i] + [1] if len(test_mask[j][i]) == max_length else [1]+test_mask[j][i]+[1] + [0] * (max_length-len(test_mask[j][i])) for i in range(len(test_mask[j]))] for j in range(len(test_mask))]


train_hist = torch.tensor(train_hist.tolist()).to(device)
train_y = torch.tensor(train_labels.tolist()).to(device)

test_hist = torch.tensor(test_hist.tolist()).to(device)
test_y = torch.tensor(test_labels.tolist()).to(device)

# freeze all the parameters
for name, param in bert.named_parameters():
    param.requires_grad = True #True

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

path = 'saved_weights_bert_'+str(max_length)+'_'+sec+'_'+bv+'_max_ep4_hist.pt' 
model.load_state_dict(torch.load(path))

# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr = 8e-4)          # learning rate

# define the loss function
mse_loss  = nn.MSELoss()  

# number of training epochs
epochs = 1

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
xs_final = None
#for each epoch
for epoch in range(epochs):

    #print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ , xs_final= train()
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), 'saved_weights_bert_'+str(max_length)+'_'+sec+'_'+bv+'_max_ep5_hist.pt')


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

#load weights of best model
path = 'saved_weights_bert_'+str(max_length)+'_'+sec+'_'+bv+'_max_ep5_hist.pt'
model.load_state_dict(torch.load(path))



# get predictions for test data

mses = []

methods = ["bare", "svr", "kr", "lr"]

xs_test,preds = test()

preds = np.asarray(preds)

test_y = test_y.cpu().data.numpy()

mse = mean_squared_error(test_y, preds)
mses.append(mse)
print("bert mse: ",mse)
lr = LinearRegression()
kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
svr = SVR(kernel='rbf', C=0.1, epsilon=0.0001) #linear')

models_list = [svr, kr, lr]

for model in models_list:
    model.fit(xs_final, train_labels.to_numpy())
    preds = model.predict(xs_test)
    mse = mean_squared_error(test_labels.to_numpy(), preds)
    mses.append(mse)
    print(model, mse)

mse = str(min(mses))+"---"+methods[mses.index(min(mses))]


#ranks  
spearmanr = (stats.spearmanr(preds, test_y))[0] 
kendallr = (stats.kendalltau(preds, test_y))[0] 

print("bert mse: ", mse)

mse_file = open('mse_bert_'+str(max_length)+'_'+sec+'_'+bv+'_max_hist.txt', "w")
mse_file.write(mse + "\n")
mse_file.write(str(spearmanr) + "\n")  
mse_file.write(str(kendallr) + "\n") 
#mse_file.close()
'''
test_error = pd.DataFrame()       
test_error['cik_year'] = test_cik.tolist()   
test_error['test_y'] = test_y.tolist()      
test_error['preds'] = [p[0] for p in preds.tolist()]  
test_error['error'] = test_error['test_y'] - test_error['preds']      
test_error.to_csv('error_bert_'+str(max_length)+'_'+sec+'_'+bv+'_max_hist.csv', index=False) 
'''

#Linear Baseline
lr = LinearRegression().fit(train_hist.cpu().data.numpy().reshape(-1, 1),
                                train_y.cpu().data.numpy().reshape(-1, 1))
preds = lr.predict(test_hist.cpu().data.numpy().reshape(-1, 1))
lr_mse = mean_squared_error(test_y.reshape(-1, 1), preds)

print("LR mse", lr_mse)
mse_file.write("Linear mse: " + str(lr_mse))
mse_file.close()


print("Total execution time: ", time.time() - start)
