import torch
from torch import nn
import torch.nn.init as init
import math

class WSDModel(nn.Module):

    def __init__(self, V, Y, D=300, dropout_prob=0.2, use_padding=False, pos_enc=None):
        super(WSDModel, self).__init__()
        self.use_padding = use_padding

        self.D = D
        self.pad_id = 0
        self.E_v = nn.Embedding(V, D, padding_idx=self.pad_id)
        self.E_y = nn.Embedding(Y, D, padding_idx=self.pad_id)
        init.kaiming_uniform_(self.E_v.weight[1:], a=math.sqrt(5))
        init.kaiming_uniform_(self.E_y.weight[1:], a=math.sqrt(5))

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))
        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([self.D])
        self.pos_enc = pos_enc

    def attention(self, X, Q, mask):
        """
        Computes the contextualized representation of query Q, given context X, using the attention model.

        :param X:
            Context matrix of shape [B, N, D]
        :param Q:
            Query matrix of shape [B, k, D], where k equals 1 (in single word attention) or N (self attention)
        :param mask:
            Boolean mask of size [B, N] indicating padding indices in the context X.

        :return:
            Contextualized query and attention matrix / vector
        """
        Q_c = None
        A = None
        
        A = torch.matmul(Q, self.W_A)
        A = torch.bmm(A, X.transpose(1, 2))

        if self.pos_enc is not None:
            encoder = self.pos_enc(A.size(1))
            A = encoder(A)
        
        if self.use_padding:
            # TODO part 2: Your code here.
            A = A.masked_fill(mask.unsqueeze(1) == self.pad_id, -math.inf)

        # TODO Part 1: continue.
        A = self.softmax(A)
        
        Q_c = torch.matmul(X, self.W_O)
        Q_c = torch.bmm(A, Q_c)

        return Q_c, A.squeeze()

    def forward(self, M_s, v_q=None):
        """
        :param M_s:
            [B, N] dimensional matrix containing token integer ids
        :param v_q:
            [B] dimensional vector containing query word indices within the sentences represented by M_s.
            This argument is only passed in single word attention mode.

        :return: logits and attention tensors.
        """

        X = self.dropout_layer(self.E_v(M_s))   # [B, N, D]
        
        Q = None
        if v_q is not None:
            # TODO Part 1: Your Code Here.
            # Look up the gather() and expand() methods in PyTorch.
            v_q = v_q.unsqueeze(1)
            v_q = v_q.unsqueeze(1)
            
            v_q = v_q.expand(len(v_q), 1, self.D)
            Q = torch.gather(X, 1, v_q)
            
        else:
            # TODO Part 3: Your Code Here.
            Q = X

        mask = M_s.ne(self.pad_id)
        Q_c, A = self.attention(X, Q, mask)
        H = self.layer_norm(Q_c + Q)
        
        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A
