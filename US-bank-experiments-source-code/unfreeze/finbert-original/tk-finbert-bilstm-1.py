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
        self.relu =  nn.ReLU()

        self.leakyrelu =  nn.LeakyReLU()

        self.elu = nn.ELU()

        self.tanh = nn.Tanh()

        self.zeros=0

        self.totals=0

        # dense layer 1
        self.fc1 = nn.Linear(768,600)
      
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(600,1)

        self.fc3 = nn.Linear(1,1)

        #LSTM
        self.hidden_dim = 768 #300
        self.emb_dim = 768
        self.encoder = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=1, bidirectional=True, dropout=0.1)


    #Define Attention Network
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden, soft_attn_weights
    

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

                del ip_id
                del attn_mask
                
                '''
                col_sum = np.sort(atten[0][0][11].sum(0)[1:-1].detach().cpu().numpy()) 
                col_sum = col_sum[::-1]  
                max_col_sum = max(col_sum) 
                top_word_mean = col_sum[:5].mean()
                chunk_max_weights.append(top_word_mean)
                '''

        #cls_vec_ = torch.mean(torch.stack(cls_vec, dim=0), dim=0)
        
        cls_vec = torch.stack(cls_vec, dim=0)
        cls_vec = cls_vec.to(torch.float32)  #LSTM
        #print("cls_vec shape: ", cls_vec.shape, type(cls_vec), cls_vec.dtype)

        '''
        x = self.fc1(cls_vec_)
        x = self.relu(x)
        x = self.dropout(x)
        

        chunk_weights = (torch.tensor(chunk_max_weights)).unsqueeze(0)
        chunk_weights = chunk_weights.cuda()    
        prod1 = torch.bmm(cls_vec.transpose(1,2), chunk_weights.transpose(0,1).unsqueeze(1)) 
        prod1 = prod1.transpose(1,2) 
        prod1 = prod1.to(torch.float32) 
        '''

        emb_input = cls_vec
        inputx = self.dropout(emb_input)
        output, (hn, cn) = self.encoder(inputx) #emb_input)
        fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] #sum bidir outputs F+B
        fbout = fbout.permute(1,0,2)
        fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
        attn_out, attn_weights = self.attnetwork(fbout, fbhn)

        '''
        chunk_weights = (torch.tensor(chunk_max_weights)).unsqueeze(0)
        chunk_weights = chunk_weights.cuda()    
        prod1 = torch.bmm(cls_vec.transpose(1,2), chunk_weights.transpose(0,1).unsqueeze(1)) 
        '''

        prod = torch.bmm(cls_vec.transpose(1,2), attn_weights.transpose(0,1).unsqueeze(1))  
        prod_sum = torch.mean(prod, 0).transpose(0,1)  

        x = prod_sum #attn_out
        
        x = self.fc1(x)
        x =self.leakyrelu(x)
        x = self.dropout(x) 

        #hist = hist.unsqueeze(0)

        #hist = self.fc3(hist)
        #x = torch.cat((x, hist.unsqueeze(0)), dim=1)
        #x = self.dropout(x)

        # output layer
        y = self.fc2(x)
        y = self.leakyrelu(y)


        return x, y


# function to train the model
def train(epoch):

    memory_file = open('memory_tk-finbert_'+str(max_length)+'_'+sec+'_'+bv+'_ep'+str(epoch)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm.txt', 'a+')
    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds = []

    total_hist = []

    xs = []


    # iterate over list of documents
    for i in range(len(train_seq)):

        memory_file.write("doc num: "+str(i)+" before train: "+str(int(torch.cuda.memory_allocated()/1024/1024))+' mem alloced\n')
        memory_file.write("doc num: "+str(i)+" before train: "+str(int(torch.cuda.memory_reserved()/1024/1024))+' mem reserved\n')

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
            #loss = huber_loss(preds, labels)
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

        loss.detach().cpu()

        memory_file.write("doc num: "+str(i)+" after train: "+str(int(torch.cuda.memory_allocated()/1024/1024))+' mem alloced\n')
        memory_file.write("doc num: "+str(i)+" after train: "+str(int(torch.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
        memory_file.flush()

        
        
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_seq)

    xs = np.array(xs)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    #total_hist = np.concatenate(total_hist, axis=0)
    memory_file.close()
    #returns the loss and predictions
    return avg_loss, total_preds , xs

# function for evaluating the model
def evaluate():

    print("\nEvaluating...")
  
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0.0, 0.0
  
    # empty list to save the model predictions
    total_preds = []

    total_xs = []

    # iterate over list of documents
    for i in range(len(valid_seq)):

        sent_id = valid_seq[i]
        mask = valid_mask[i]
        hist = valid_hist[i]
        labels = valid_y[i].unsqueeze(0).unsqueeze(0)

        # deactivate autograd
        with torch.no_grad():
      
            with autocast():
            # model predictions
                x, preds = model(sent_id, mask, hist)
                
                # compute the validation loss between actual and predicted values
                loss = mse_loss(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

            x = x.detach().cpu().numpy().ravel()

            total_xs.append(x)

        loss.detach().cpu()

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(valid_seq) 

    total_xs = np.array(total_xs)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, total_xs

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
            
            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

            x = x.detach().cpu().numpy().ravel()

            total_xs.append(x)

        
            
    # reshape the predictions in form of (number of samples, no. of classes)
    total_xs = np.array(total_xs)

    total_preds = np.concatenate(total_preds, axis=0)
    
    return total_xs, total_preds

def train_x():

    # empty list to save the model predictions
    total_xs = []

    total_preds=[]
    

    for i in range(len(train_seq)):

        sent_id = train_seq[i]
        mask = train_mask[i]
        hist = train_hist[i]
        #labels = test_y[i].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            with autocast():
                x, preds = model(sent_id, mask, hist)
            
            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

            x = x.detach().cpu().numpy().ravel()

            total_xs.append(x)

        
            
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

train_text, rem_text, train_hist, rem_hist, train_labels, rem_labels = train_test_split(df['mda'],
    df['prev_'+bv], 
    df[bv],
    shuffle=False,
    train_size=0.8) 

valid_text, test_text, valid_hist, test_hist, valid_labels, test_labels = train_test_split(
	rem_text,
	rem_hist,
	rem_labels,
    shuffle=False,
	test_size=0.5
)

'''
val_text, test_text, val_hist, test_hist, val_labels, test_labels = train_test_split(temp_text, 
    temp_hist,
        temp_labels,
            shuffle=False,
                test_size=0.2) 

val_text = val_text.astype(str)
'''

train_text = train_text.astype(str) 
valid_text = valid_text.astype(str)
test_text = test_text.astype(str)

'''
df_train = pd.DataFrame()
df_test = pd.DataFrame()

for y in train_years_list:
    df_train = pd.concat([df_train, pd.read_csv(str(y) + "_tok.csv")])
'''

#bert_path = "/gpfs/u/home/HPDM/HPDMrawt/scratch/npl_env/sdm21-exps/long_document_fin/"

bert_path = "/gpfs/u/home/DLTM/DLTMboxi/scratch/env/finbert/"

config = AutoConfig.from_pretrained(bert_path, output_attentions=True) 

# import BERT-base pretrained model
bert = AutoModel.from_pretrained(bert_path, config=config) #longformer-base-4096/') 

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(bert_path) #longformer-base-4096/') 

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

#VALID
# tokenize and encode sequences in the training set
tokens_valid = tokenizer.batch_encode_plus(
    valid_text.tolist(),
    add_special_tokens=False
)

#Extract input ids
valid_seq_ = tokens_valid['input_ids']
#Split each document into 510 tokens
valid_seq = [[valid_seq_[j][i:i + max_length] for i in range(0, len(valid_seq_[j]), max_length)] for j in range(len(valid_seq_))]
#print(valid_seq[0][0])
#Add [CLS], [SEP] and [PAD] tokens
valid_seq = [[[tokenizer.cls_token_id] + valid_seq[j][i] + [tokenizer.sep_token_id] if len(valid_seq[j][i]) == max_length else [tokenizer.cls_token_id] + valid_seq[j][i] +[tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (max_length-len(valid_seq[j][i])) for i in range(len(valid_seq[j]))] for j in range(len(valid_seq))]
#print(valid_seq[0][0])
#df_valid_seq=pd.DataFrame()
#df_valid_seq["valid_seq"]=valid_seq
#df_valid_seq.to_csv(sec+ "-valid_seq.csv")

#Extract attention masks
valid_mask_ = tokens_valid['attention_mask']
#Split each document into 510 tokens
valid_mask = [[valid_mask_[j][i:i + max_length] for i in range(0, len(valid_mask_[j]), max_length)] for j in range(len(valid_mask_))]
#Add [1] for attention and [0] for [PAD]
valid_mask = [[[1] + valid_mask[j][i] + [1] if len(valid_mask[j][i]) == max_length else [1]+valid_mask[j][i]+[1] + [0] * (max_length-len(valid_mask[j][i])) for i in range(len(valid_mask[j]))] for j in range(len(valid_mask))]

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

valid_hist = torch.tensor(valid_hist.tolist()).to(device)
valid_y = torch.tensor(valid_labels.tolist()).to(device)

test_hist = torch.tensor(test_hist.tolist()).to(device)
test_y = torch.tensor(test_labels.tolist()).to(device)

#val_hist = torch.tensor(val_hist.tolist()).to(device)
#val_y = torch.tensor(val_labels.tolist()).to(device)

# freeze all the parameters
for name, param in bert.named_parameters():
    param.requires_grad = True #True

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

# define the loss function
mse_loss  = nn.MSELoss()  
huber_loss  = nn.L1Loss()

# number of training epochs
total_epochs = int(sys.argv[4])
start_epoch = int(sys.argv[5])
end_epoch = int(sys.argv[6])
epochs = end_epoch - start_epoch + 1
#plus = int(sys.argv[5])

# different learning rates
learning_rate = float(sys.argv[7])

# set initial loss to previous best
best_valid_loss = float('inf')
best_epoch = 0
# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):

    #print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    torch.cuda.empty_cache()
    # define the optimizer
    optimizer = AdamW(model.parameters(),
                lr = learning_rate, eps = 1e-8)          # learning rate
    
    
    #train model
    train_loss, _ , xs_final= train(start_epoch+epoch)
    
    #evaluate model
    valid_loss, _ , _ = evaluate()

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = start_epoch + epoch
        #print(f'\nTraining Loss: {train_loss:.3f}')
        #xs_train = xs_final
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), 'saved_weights_tk-finbert_'+str(max_length)+'_'+sec+'_'+bv+'_ep'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm.pt')
        #torch.save(model_to_save.state_dict(), 'saved_weights_tk-finbert_'+str(max_length)+'_'+sec+'_'+bv+'_epoch'+str(start_epoch+epoch)+'_of_'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm.pt')

    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
            
    print(f'\nTraining Loss: {train_loss:.10f}')
    print(f'Validation Loss: {valid_loss:.10f}')

valid_loss_file = open('best_valid_loss_tk-finbert_'+str(max_length)+'_'+sec+'_'+bv+'_ep'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm.txt', 'w')
valid_loss_file.write(str(best_valid_loss)+"\n")
valid_loss_file.write(str(best_epoch))
valid_loss_file.close()
'''
# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

#load weights of best model
path = 'saved_weights_tk-finbert_'+str(max_length)+'_'+sec+'_'+bv+'_ep'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm.pt'
model.load_state_dict(torch.load(path))

xs_train , _ = train_x()

# get predictions for test data
valid_mses = []
test_mses = []

methods = ["bare", "svr", "kr", "lr"]

_ , preds, xs_valid = evaluate()
preds = np.asarray(preds)
valid_y = valid_y.cpu().data.numpy()
valid_mse = mean_squared_error(valid_y, preds)
valid_mses.append(valid_mse)

xs_test, preds = test()
preds = np.asarray(preds)
test_y = test_y.cpu().data.numpy()
test_mse = mean_squared_error(test_y, preds)
test_mses.append(test_mse)

print("bert mse: ",test_mse)
lr = LinearRegression()
kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
svr = SVR(kernel='rbf', C=0.1, epsilon=0.0001) #linear')

models_list = [svr, kr, lr]

for m in models_list:
    m.fit(xs_train, train_labels.to_numpy())

    preds = m.predict(xs_valid)
    valid_mse = mean_squared_error(valid_labels.to_numpy(), preds)
    valid_mses.append(valid_mse)

    preds = m.predict(xs_test)
    test_mse = mean_squared_error(test_labels.to_numpy(), preds)
    test_mses.append(test_mse)
    print(m, test_mse,'---',valid_mse)


mse = str(test_mses[valid_mses.index(min(valid_mses))])+"---"+methods[valid_mses.index(min(valid_mses))]+"---"+str(min(valid_mses))


spearmanr = (stats.spearmanr(preds, test_y))[0] 
kendallr = (stats.kendalltau(preds, test_y))[0]  

print("tk-finbert mse: ", mse)

mse_file = open('mse_tk-finbert_'+str(max_length)+'_'+sec+'_'+bv+'_ep'+str(epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_bilstm.txt', "w")
mse_file.write(mse + "\n")
mse_file.write(str(best_valid_loss)+"\n")
mse_file.write(str(spearmanr) + "\n")    
mse_file.write(str(kendallr) + "\n")       
#mse_file.close()

test_error = pd.DataFrame()               
test_error['cik_year'] = test_cik.tolist()   
test_error['test_y'] = test_y.tolist()       
test_error['preds'] = [p[0] for p in preds.tolist()]    
test_error['error'] = test_error['test_y'] - test_error['preds']            
test_error.to_csv('error_tk-finbert_'+str(max_length)+'_'+sec+'_'+bv+'_mean_hist.csv', index=False)          


#Linear Baseline
lr = LinearRegression().fit(train_hist.cpu().data.numpy().reshape(-1, 1),
                                train_y.cpu().data.numpy().reshape(-1, 1))
preds = lr.predict(test_hist.cpu().data.numpy().reshape(-1, 1))
lr_mse = mean_squared_error(test_y.reshape(-1, 1), preds)

print("LR mse", lr_mse)
mse_file.write("Linear mse: " + str(lr_mse))
mse_file.close()
'''

print("Total execution time: ", time.time() - start)
