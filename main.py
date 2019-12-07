import torch                                     
import torch.nn as nn                           
from torch.autograd import Variable            
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models
import timeit
from dataloader import DeepSeqDataset
from model import Resound, action_branch
from opts import parse_opts
import os
import numpy as np

########################## Initializing the dataloader
opt = parse_opts()

if opt.root_path != '':
	path_model = os.path.join(opt.root_path,opt.result_path) # Path to saved model
	frames_dir_train = os.path.join(opt.video_path,'rgb_frames_processed_all_frames/') # Path to train images
	frames_dir_test = os.path.join(opt.video_path,'test_all_frames/') # Path to test images

train_data_48 = np.load(os.path.join(opt.root_path,'train_data_64.npy'))
test_data_48 = np.load(os.path.join(opt.root_path,'test_data_64.npy'))
vocab= np.load(os.path.join(opt.root_path,'vocab.npy'))

dataloader = {'train': DataLoader(DeepSeqDataset(frames_dir_train,vocab,train_data_48,opt.sample_duration,'train'), batch_size= opt.batch_size, shuffle=True, num_workers=opt.n_threads,pin_memory=True),
			   'test': DataLoader(DeepSeqDataset(frames_dir_test,vocab,test_data_48,opt.sample_duration,'test'), batch_size= opt.batch_size, shuffle=True, num_workers=opt.n_threads,pin_memory=True)}

dataset_sizes = {'train':len(DeepSeqDataset(frames_dir_train,vocab,train_data_48,opt.sample_duration,'train')),
				 'test': len(DeepSeqDataset(frames_dir_test,vocab,test_data_48,opt.sample_duration,'test'))}

revert= np.array(range(opt.sample_duration-1,-1,-1))
# In[86]:

########################## Initializing the model

model_conv = torchvision.models.resnet18(pretrained=True)

for param in model_conv.parameters():
	param.requires_grad = True
	
modules = list(model_conv.children())[:-2]      # delete last fc layer.
model_conv_new = nn.Sequential(*modules)

if not os.path.exists(os.path.join(path_model,'models_{}_{}_{}_{}'.format(opt.input_size_encoder_attn,opt.hidden_size,opt.hidden_size_decoder,opt.num_layers))):
	os.makedirs(os.path.join(path_model,'models_{}_{}_{}_{}'.format(opt.input_size_encoder_attn,opt.hidden_size,opt.hidden_size_decoder,opt.num_layers)))

resound = Resound(model_conv_new, opt.input_size_encoder_attn, opt.hidden_size, opt.num_layers,opt.sample_duration,opt.hidden_size_decoder,revert)

model_branched = action_branch(opt.hidden_size_decoder,opt.sample_duration)

if opt.resume_epoch >=0:
	PATH_model= os.path.join(path_model,'models_{}_{}_{}_{}/epoch_{:05d}_model.pth.tar'.format(opt.input_size_encoder_attn,opt.hidden_size,opt.hidden_size_decoder,opt.num_layers,opt.resume_epoch))
	PATH_branched= os.path.join(path_model,'models_{}_{}_{}_{}/epoch_{:05d}_model_branched.pth.tar'.format(opt.input_size_encoder_attn,opt.hidden_size,opt.hidden_size_decoder,opt.num_layers,opt.resume_epoch))

	resound.load_state_dict(torch.load(PATH_model,opt.resume_path))
	model_branched.load_state_dict(torch.load(PATH_branched))
 
if not opt.no_cuda:
	resound = resound.cuda()
	model_branched = model_branched.cuda()
	
print_every_batch=50

resound_optimizer = optim.SGD(resound.parameters(), lr = opt.learning_rate, momentum=opt.momentum, nesterov=True)
model_branched_optimizer = optim.SGD(model_branched.parameters(), lr = opt.learning_rate, momentum=opt.momentum, nesterov=True)

criterion = nn.CrossEntropyLoss().cuda() if not opt.no_cuda else nn.CrossEntropyLoss()


# Training and Testing

for epoch in range(1,opt.n_epochs+1):
	print('Epoch {}/{}'.format(epoch, opt.n_epochs - 1))
	print('-' * 10)
		
	# Each epoch has a training and validation phase
	for phase in ['train','test']:
		start_time = timeit.default_timer()
		if phase == 'train':
			resound.train(True)
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
			label = label.cuda() if not opt.no_cuda else label

			action = torch.LongTensor(input_x['action'].numpy())
			action = action.cuda() if not opt.no_cuda else action

			batch_size = input_variable.shape[0]

			resound_optimizer.zero_grad()
			model_branched_optimizer.zero_grad()

			loss = Variable(torch.zeros(1,1))
			loss = loss.cuda() if not opt.no_cuda else loss
			
			input_seq_var = Variable(input_variable);
			input_seq_var = input_seq_var.cuda() if not opt.no_cuda else input_seq_var

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
			
			if opt.repre_learning==0:
				loss_final = loss
			else:
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
				print("\t[{}] Epoch: {}/{} Batch: {} Loss: {:0.4f} Acc: {:0.4f} Acc_0: {:0.4f} Acc_1: {:0.4f} Acc_2: {:0.4f} Acc_3: {:0.4f}".format(phase, epoch, opt.n_epochs, i, batch_loss, batch_corrects.double()/batch_size, batch_corrects_0.double()/batch_size, batch_corrects_1.double()/batch_size, batch_corrects_2.double()/batch_size, batch_corrects_3.double()/batch_size))

		if phase == 'train':
			torch.save(resound.module.state_dict(),path_model + 'models_{}_{}_{}_{}/epoch_{:05d}_model.pth.tar'.format(opt.input_size_encoder_attn,opt.hidden_size,opt.hidden_size_decoder,opt.num_layers,epoch))
			torch.save(model_branched.module.state_dict(),path_model + 'models_{}_{}_{}_{}/epoch_{:05d}_model_branched.pth.tar'.format(opt.input_size_encoder_attn,opt.hidden_size,opt.hidden_size_decoder,opt.num_layers,epoch))
			print('model saved')

		epoch_loss = running_loss*batch_size / dataset_sizes[phase]
		epoch_acc = running_corrects.double() / dataset_sizes[phase]
		epoch_acc_0 = running_corrects_0.double() / dataset_sizes[phase]
		epoch_acc_1 = running_corrects_1.double() / dataset_sizes[phase]
		epoch_acc_2 = running_corrects_2.double() / dataset_sizes[phase]
		epoch_acc_3 = running_corrects_3.double() / dataset_sizes[phase]

		print("[{}] Epoch: {}/{} Loss: {:0.4f} Acc: {:0.4f} Acc_0: {:0.4f} Acc_1: {:0.4f} Acc_2: {:0.4f} Acc_3: {:0.4f}".format(phase, epoch, opt.n_epochs, epoch_loss, epoch_acc, epoch_acc_0, epoch_acc_1, epoch_acc_2, epoch_acc_3))
		stop_time = timeit.default_timer()
		print("Execution time: " + str(stop_time - start_time) + "\n")

