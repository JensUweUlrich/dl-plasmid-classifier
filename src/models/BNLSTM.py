import torch

from torch import nn
import lightning.pytorch as pl
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
from torch.nn.init import kaiming_normal, kaiming_uniform, constant

class SeparatedBatchNorm1d(nn.Module):
	
    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-3, momentum=0.1,
				 affine=True):
        """
		Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
				'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
				'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
							 .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm(
			input=input_, running_mean=running_mean, running_var=running_var,
			weight=self.weight, bias=self.bias, training=self.training,
			momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
				' max_length={max_length}, affine={affine})'
				.format(name=self.__class__.__name__, **self.__dict__))

class BNLSTMCell(nn.Module):

    """A BN-LSTM cell."""
	
    def __init__(self, input_size, hidden_size, max_length, use_bias=True):

        super(BNLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
			torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
			torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
		# BN parameters
        self.bn_ih = SeparatedBatchNorm1d(
			num_features=4 * hidden_size, max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(
	    	num_features=4 * hidden_size, max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(
			num_features=hidden_size, max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
		Initialize parameters following the way proposed in the paper.
		"""

        init.orthogonal_(self.weight_ih.data)
		# The hidden-to-hidden weight matrix is initialized as an identity
		# matrix.
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
		# The bias is just set to zero vectors.
        init.constant_(self.bias.data, val=0)
		# Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        """
		Args:
			input_: A (batch, input_size) tensor containing input
				features.
			hx: A tuple (h_0, c_0), which contains the initial hidden
				and cell state, where the size of both states is
				(batch, hidden_size).
			time: The current timestep value, which is used to
				get appropriate running statistics.
		Returns:
			h_1, c_1: Tensors containing the next hidden and cell state.
		"""

		# print(input_)
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
					  .expand(batch_size, *self.bias.size()))
        wh = torch.mm(h_0, self.weight_hh)
		# print(input_, self.weight_ih, time)
        wi = torch.mm(input_, self.weight_ih)
        bn_wh = self.bn_hh(wh, time=time)
        bn_wi = self.bn_ih(wi, time=time)
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,
								 split_size_or_sections=self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))
		# print(h_1)
        return h_1, c_1


class bnLSTM(pl.LightningModule):

    """A module that runs multiple steps of LSTM."""

	# def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
    def __init__(self, input_size, hidden_size,max_length, learning_rate, 
                 batch_size, train_pos_weight, val_pos_weight, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0.5, num_classes = 1):
        super(bnLSTM, self).__init__()
		# self.cell_class = cell_class
        self.real_input_size = input_size
        self.input_size = input_size

        self.lr = learning_rate
        self.batch_size = batch_size
        self.train_criterion = nn.BCEWithLogitsLoss(pos_weight=train_pos_weight)
        self.val_criterion = nn.BCEWithLogitsLoss(pos_weight=val_pos_weight)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
		# self.seqLength = seqLength
        self.fc = nn.Linear(in_features=hidden_size, out_features=int(num_classes))
        for layer in range(num_layers):
            layer_input_size = self.input_size  if layer == 0 else hidden_size
            cell = BNLSTMCell(input_size=layer_input_size,
							  hidden_size=hidden_size, max_length = max_length, use_bias=True)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# if isinstance(cell, BNLSTMCell):
			# 	h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
			# else:
			# 	h_next, c_next = cell(input_=input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
		# print( output, hx)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)


        h0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
        c0 = Variable(input_.data.new(input_.size(1), self.hidden_size)
					  .normal_(0, 0.1))
        hx = (h0, c0)
        #print(input_.size())
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            if layer == 0:
                layer_output, (layer_h_n, layer_c_n) = bnLSTM._forward_rnn(
			        cell=cell, input_=input_, length=length, hx=hx)
            else:
                layer_output, (layer_h_n, layer_c_n) = bnLSTM._forward_rnn(
					cell=cell, input_=layer_output, length=length, hx=hx)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)


        output = layer_output
        output = output[-1]
		
		# Note JUU 18/10/2023
		# Use FC Layer with only one outpute node as final Layer when using BCEWithLogitLoss
		# Internally uses sigmoid for binary classification
		# softmax only suitable for multi-class problems
		#output = functional.softmax(self.fc(output), dim=1)
        output = self.fc(output)

        return output
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
     
    def training_step(self, train_batch, train_batch_idx):
        data, labels, ids = train_batch
        train_labels = labels.to(torch.float)
        #print(data.shape)
        outputs_train = self.forward(data)
        train_loss = self.train_criterion(outputs_train, labels.unsqueeze(1))
        pred_labels = (outputs_train >= 0.5).type(torch.float)
        train_acc = (train_labels == pred_labels).float().mean().item()
        self.log('train_loss', train_loss, batch_size=self.batch_size, )
        self.log('train_acc', train_acc, batch_size=self.batch_size)
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        data, labels, val_ids = val_batch
        #print(data.shape)
        val_outputs = self.forward(data)
        val_loss = self.val_criterion(val_outputs, labels.unsqueeze(1).to(torch.float))
        predicted_labels = (val_outputs >= 0.5).type(torch.long).data.numpy()
        val_labels = labels.to(torch.long).numpy()
        val_acc = balanced_accuracy_score(val_labels, predicted_labels)
        self.log('val_loss', val_loss, batch_size=self.batch_size)
        self.log('val_bal_acc', val_acc, batch_size=self.batch_size)