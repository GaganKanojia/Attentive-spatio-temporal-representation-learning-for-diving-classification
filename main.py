import torch                                     
import torch.nn as nn                           
from torch.autograd import Variable            
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models
import timeit
from dataloader import DeepSeqDataset
from model import Resound, action_branch
use_cuda = torch.cuda.is_available()


########################## Initializing the dataloader
batch_size = 4
seq_len = 64
path = '/home/parker/sudhakar/Resound_action_final/' # Path to root folder
path_model = path+'models_lstm_64_augumented_test/' # Path to saved model
frames_dir_train = path+'rgb_frames_processed_all_frames/' # Path to train images
frames_dir_test = path+'test_all_frames/' # Path to test images
repre_learning = 0 # 1 for representation learning, 0 for otherwise
train_data_48 = np.load(path+'train_data_64.npy')
test_data_48 = np.load(path+'test_data_64.npy')

vocab= np.load(path+'vocab.npy')

dataloader = {'train': DataLoader(DeepSeqDataset(frames_dir_train,vocab,train_data_48,seq_len,'train'), batch_size= batch_size, shuffle=True, num_workers=2,pin_memory=True),
			   'test': DataLoader(DeepSeqDataset(frames_dir_test,vocab,test_data_48,seq_len,'test'), batch_size= 2, shuffle=True, num_workers=2,pin_memory=True)}

dataset_sizes = {'train':len(DeepSeqDataset(frames_dir_train,vocab,train_data_48,seq_len,'train')),
				 'test': len(DeepSeqDataset(frames_dir_test,vocab,test_data_48,seq_len,'test'))}

revert= np.array(range(seq_len-1,-1,-1))
# In[86]:

########################## Initializing the model

model_conv = torchvision.models.resnet18(pretrained=True)

for param in model_conv.parameters():
	param.requires_grad = True
	
modules = list(model_conv.children())[:-2]      # delete last fc layer.
model_conv_new = nn.Sequential(*modules)

input_size_lstm_encoder = 1024
hidden_size = 512
hidden_size_decoder_2 = 256
num_layers=3
out_size = seq_len

if not os.path.exists(os.path.join(path_model,'models_{}_{}_{}_{}'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers))):
	os.makedirs(os.path.join(path_model,'models_{}_{}_{}_{}'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers)))

PATH_model= path_model + 'models_{}_{}_{}_{}/epoch_00042_model.pth.tar'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers)
PATH_branched= path_model + 'models_{}_{}_{}_{}/epoch_00042_model_branched.pth.tar'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers)

load_model = 1
dual = 0

resound = Resound(model_conv_new, input_size_lstm_encoder, hidden_size, num_layers,seq_len,hidden_size_decoder_2,revert)
model_branched = action_branch(hidden_size_decoder_2,seq_len)

if load_model ==1:
	resound.load_state_dict(torch.load(PATH_model))
	model_branched.load_state_dict(torch.load(PATH_branched))

if use_cuda:
	resound = resound.cuda()
	model_branched = model_branched.cuda()
	
num_epochs=100
print_every_batch=50

plot_losses = []
print_loss_total = 0
plot_loss_total = 0

resound_optimizer = optim.SGD(resound.parameters(), lr = 0.1, momentum=0.9, nesterov=True)
model_branched_optimizer = optim.SGD(model_branched.parameters(), lr = 0.1, momentum=0.9, nesterov=True)

criterion = nn.CrossEntropyLoss().cuda() if use_cuda else nn.CrossEntropyLoss()


# Training and Testing

for epoch in range(1,num_epochs+1):
	print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	print('-' * 10)
		
	# Each epoch has a training and validation phase
	for phase in ['train','test']:
	#for phase in ['train']:		
		start_time = timeit.default_timer()
		if phase == 'train':
			resound.train(True)
			#resound.eval()
			model_branched.train(True)
		elif phase == 'test':
			resound.eval()
			model_branched.eval()

		running_loss=0.0
		running_corrects=0.0
		running_corrects_0=0.0
		running_corrects_1=0.0
		running_corrects_2=0.0
		running_corrects_3=0.0

		for i,input_x in enumerate(dataloader[phase]):

			batch_corrects=0.0
			batch_corrects_0=0.0
			batch_corrects_1=0.0
			batch_corrects_2=0.0
			batch_corrects_3=0.0

			input_variable = input_x['vid_frames']

			label = torch.LongTensor(input_x['label'].numpy())
			label = label.cuda() if use_cuda else label

			action = torch.LongTensor(input_x['action'].numpy())
			action = action.cuda() if use_cuda else action

			batch_size = input_variable.shape[0]

			resound_optimizer.zero_grad()
			model_branched_optimizer.zero_grad()

			loss = Variable(torch.zeros(1,1))
			loss = loss.cuda() if use_cuda else loss
			
			input_seq_var = Variable(input_variable);
			input_seq_var = input_seq_var.cuda() if use_cuda else input_seq_var

			logits = resound.forward(input_seq_var)
			output,output_branched0,output_branched1,output_branched2,output_branched3 = model_branched.forward(logits)
			
			probs = nn.Softmax(dim=1)(output)
			preds = torch.max(probs, 1)[1]
			loss = criterion(output, label)

			probs0 = nn.Softmax(dim=1)(output_branched0)
			preds0 = torch.max(probs0, 1)[1]
			loss0 = criterion(output_branched0, action[:,0])

			probs1 = nn.Softmax(dim=1)(output_branched1)
			preds1 = torch.max(probs1, 1)[1]
			loss1 = criterion(output_branched1, action[:,1])

			probs2 = nn.Softmax(dim=1)(output_branched2)
			preds2 = torch.max(probs2, 1)[1]
			loss2 = criterion(output_branched2, action[:,2])

			probs3 = nn.Softmax(dim=1)(output_branched3)
			preds3 = torch.max(probs3, 1)[1]
			loss3 = criterion(output_branched3, action[:,3])
			
			if repre_learning==0:
				loss_final = loss
			elif repre_learning==0:
				loss_final = loss0 + loss1 + loss2 + loss3

			if phase == 'train':
				loss_final.backward()
				resound_optimizer.step()
				model_branched_optimizer.step()

			batch_loss = loss_final.item() * input_seq_var.size(0)
			batch_corrects = torch.sum(preds == label)
			batch_corrects_0 = torch.sum(preds0 == action[:,0])
			batch_corrects_1 = torch.sum(preds1 == action[:,1])
			batch_corrects_2 = torch.sum(preds2 == action[:,2])
			batch_corrects_3 = torch.sum(preds3 == action[:,3])

			running_loss += loss_final.item() * input_seq_var.size(0)
			running_corrects += batch_corrects
			running_corrects_0 += batch_corrects_0
			running_corrects_1 += batch_corrects_1
			running_corrects_2 += batch_corrects_2
			running_corrects_3 += batch_corrects_3
			
			if i%print_every_batch==0:
				print("\t[{}] Epoch: {}/{} Batch: {} Loss: {:0.4f} Acc: {:0.4f} Acc_0: {:0.4f} Acc_1: {:0.4f} Acc_2: {:0.4f} Acc_3: {:0.4f}".format(phase, epoch, num_epochs, i, batch_loss, batch_corrects.double()/batch_size, batch_corrects_0.double()/batch_size, batch_corrects_1.double()/batch_size, batch_corrects_2.double()/batch_size, batch_corrects_3.double()/batch_size))

		if phase == 'train' and dual==0:
			torch.save(resound.state_dict(),path_model + 'models_{}_{}_{}_{}/epoch_{:05d}_model.pth.tar'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers,epoch))
			torch.save(model_branched.state_dict(),path_model + 'models_{}_{}_{}_{}/epoch_{:05d}_model_branched.pth.tar'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers,epoch))
			print('model saved')
		elif phase == 'train' and dual==1:
			torch.save(resound.module.state_dict(),path_model + 'models_{}_{}_{}_{}/epoch_{:05d}_model.pth.tar'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers,epoch))
			torch.save(model_branched.module.state_dict(),path_model + 'models_{}_{}_{}_{}/epoch_{:05d}_model_branched.pth.tar'.format(input_size_lstm_encoder,hidden_size,hidden_size_decoder_2,num_layers,epoch))
			print('model saved')

		epoch_loss = running_loss*batch_size / dataset_sizes[phase]
		epoch_acc = running_corrects.double() / dataset_sizes[phase]
		epoch_acc_0 = running_corrects_0.double() / dataset_sizes[phase]
		epoch_acc_1 = running_corrects_1.double() / dataset_sizes[phase]
		epoch_acc_2 = running_corrects_2.double() / dataset_sizes[phase]
		epoch_acc_3 = running_corrects_3.double() / dataset_sizes[phase]

		print("[{}] Epoch: {}/{} Loss: {:0.4f} Acc: {:0.4f} Acc_0: {:0.4f} Acc_1: {:0.4f} Acc_2: {:0.4f} Acc_3: {:0.4f}".format(phase, epoch, num_epochs, epoch_loss, epoch_acc, epoch_acc_0, epoch_acc_1, epoch_acc_2, epoch_acc_3))
		stop_time = timeit.default_timer()
		print("Execution time: " + str(stop_time - start_time) + "\n")

