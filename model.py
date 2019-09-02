import torch     
import torch.nn as nn
                       
class Resound(nn.Module):
	def __init__(self,model_conv_new, input_size_lstm, hidden_size, num_layers,seq_len,hidden_size_2,revert):
		super(Resound, self).__init__()
		self.model = model_conv_new
		self.conv1 = nn.Conv2d(512, input_size_lstm, 1, stride=1, padding=0)
		self.avgpool1 = nn.AvgPool2d(7)

		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm_encoder = nn.LSTM(input_size_lstm, hidden_size, num_layers, batch_first=True)
		self.seq_len = seq_len
		self.lstm_attention = nn.LSTM(input_size_lstm, hidden_size, num_layers, batch_first=True)
		self.lstm_decoder = nn.LSTM(input_size_lstm, hidden_size_2, num_layers, batch_first=True)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.5)

		self.linear_hidden_h0 = nn.Linear(hidden_size, hidden_size_2, bias=True)
		self.linear_hidden_c0 = nn.Linear(hidden_size, hidden_size_2, bias=True)
		self.linear0 = nn.Linear(hidden_size, input_size_lstm, bias=True)

		self.revert = revert

		for name, param in self.lstm_encoder.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.xavier_normal_(param)

		for name, param in self.lstm_attention.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.xavier_normal_(param)

		for name, param in self.lstm_decoder.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.xavier_normal_(param)

	def forward(self,x_init):

		x = x_init.view(x_init.size(0)*x_init.size(1),x_init.size(2),x_init.size(3),x_init.size(4))
		x = self.model(x)
		x = self.conv1(x)
		x = self.avgpool1(x)
		x = x.view(x_init.size(0),x_init.size(1),-1)

		##### Encoder
		_, hidden_encoder = self.lstm_encoder(x)

		##### Attention
		x = x[:,self.revert,:]
		out, _ = self.lstm_attention(x.contiguous(), hidden_encoder)
		out = self.relu(self.linear0(self.dropout(out)))
		out = torch.mul(out,x)

		##### Decoder
		h0 = self.linear_hidden_h0(hidden_encoder[0])
		c0 = self.linear_hidden_c0(hidden_encoder[1])
		
		out, _ = self.lstm_decoder(out, (h0,c0))

		return out

class action_branch(nn.Module):

	def __init__(self, hidden_size,seq_len):
		super(action_branch, self).__init__()

		self.time_step_weight_action = nn.Parameter(torch.rand(1,seq_len,1))
		self.time_step_weight_fc0 = nn.Parameter(torch.rand(1,seq_len,1))
		self.time_step_weight_fc1 = nn.Parameter(torch.rand(1,seq_len,1))
		self.time_step_weight_fc2 = nn.Parameter(torch.rand(1,seq_len,1))
		self.time_step_weight_fc3 = nn.Parameter(torch.rand(1,seq_len,1))

		self.action = nn.Linear(hidden_size, 48, bias=True)
		self.fc0 = nn.Linear(hidden_size, 4)
		self.fc1 = nn.Linear(hidden_size, 8)
		self.fc2 = nn.Linear(hidden_size, 8)
		self.fc3 = nn.Linear(hidden_size, 4)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(p=0.2)

	def forward(self, h):

		h = self.dropout(h)
		
		out = self.action(h)
		out = torch.sum(out*self.time_step_weight_action.expand_as(out),dim=1)

		logits0 = self.fc0(h)
		logits0 = torch.sum(logits0*self.time_step_weight_fc0.expand_as(logits0),dim=1)
		logits1 = self.fc1(h)
		logits1 = torch.sum(logits1*self.time_step_weight_fc1.expand_as(logits1),dim=1)
		logits2 = self.fc2(h)
		logits2 = torch.sum(logits2*self.time_step_weight_fc2.expand_as(logits2),dim=1)
		logits3 = self.fc3(h)
		logits3 = torch.sum(logits3*self.time_step_weight_fc3.expand_as(logits3),dim=1)

		return out,logits0,logits1,logits2,logits3


