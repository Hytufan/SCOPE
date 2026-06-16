import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)

class AttrProxy(object):
    """
    In PyTorch it does not currently work to have an array of variables in a
    module. These parameters will not be included in the overall module's
    parameter list. To get around this you need manually add them to the
    module. This is then a helper to access the particular sub-module from the
    "list"
    See https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    for more disucssion.
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class PropModel(nn.Module):
    def __init__(self, state_dim, n_nodes, n_edge_types):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_edge_types = n_edge_types
        self.reset_gate = nn.Sequential(
                nn.Linear(state_dim * 3, state_dim),
                nn.Sigmoid()
            )

        self.update_gate = nn.Sequential(
                nn.Linear(state_dim * 3, state_dim),
                nn.Sigmoid()
            )

        self.transform = nn.Sequential(
                nn.Linear(state_dim * 3, state_dim),
                nn.Tanh()
            )

    def forward(self, state_in, state_out, state_cur, adj_matrix):
        # state_in & state_out: [batch_size, n_nodes * n_edge_types, state_dim]
        A_in = adj_matrix[:, :, :self.n_nodes * self.n_edge_types]  # A_in: [batch_size, n_nodes, n_nodes * n_edge_types]
        A_out = adj_matrix[:, :, self.n_nodes * self.n_edge_types:] # A_out: [batch_size, n_nodes, n_nodes * n_edge_types]

        # bmm is a Batch matrix multiplication
        a_in = torch.bmm(A_in, state_in)            # a_in: [batch_size, n_nodes, state_dim]
        a_out = torch.bmm(A_out, state_out)         # a_out: [batch_size, n_nodes, state_dim]

        # Equation 2 of section 3.2 in the paper
        # Which nodes can pass information to other nodes between edges.
        a = torch.cat((a_in, a_out, state_cur), 2)  # a: [batch_size, n_nodes, state_dim * 3]

        # Equation 3 of section 3.2
        z = self.update_gate(a)                     # z: [batch_size, n_nodes, state_dim]

        # Equation 4 of section 3.2
        r = self.reset_gate(a)                      # r: [batch_size, n_nodes, state_dim]

        # Equation 5  of section 3.2
        h_hat = self.transform(torch.cat((a_in, a_out, r * state_cur), 2))

        # Equation 6 of section 3.2
        output = (1 - z) * state_cur + z * h_hat

        return output

class AttentionLayer(nn.Module):
    def __init__(self, in_state_dim, out_state_dim, alpha=0, dropout=0.2, concat=False):
        super().__init__()
        self.in_state_dim = in_state_dim
        self.out_state_dim = out_state_dim
        self.concat = concat

        self.w = nn.Linear(self.in_state_dim, self.out_state_dim)
        self.a_i = nn.Linear(self.out_state_dim, 1)
        self.a_j = nn.Linear(self.out_state_dim, 1)
        # self.a = nn.Parameter(torch.empty(size=(2 * out_state_dim, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

        self._initialize_weights()
        

    def forward(self, in_hidden_state, adj_matrix):
        batch_size, n_nodes, _ = in_hidden_state.shape
        # print("input state:", in_hidden_state)
        Wh = self.w(in_hidden_state)
        Wh_i = self.a_i(Wh)
        Wh_j = self.a_j(Wh)
        # print("Wh_i, Wh_j:", Wh_i, Wh_j)
        e = Wh_i + Wh_j.transpose(1, 2)
        e = self.leakyrelu(e)
        # print("e:", e, e.shape)
        zero = -9e15 * torch.ones_like(e)
        attn = torch.where(adj_matrix > 0, e, zero)
        # print("attn1:", attn, attn.shape)
        attn = self.softmax(attn)
        # print("attn2:", attn, attn.shape)
        attn = self.dropout(attn)
        # print("attn3:", attn, attn.shape)
        out_hidden_state = torch.bmm(attn, Wh)
        # print("output state:", out_hidden_state)
        if self.concat:
            return self.activation(out_hidden_state)
        else:
            return out_hidden_state

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_state_dim) + ' -> ' + str(self.out_state_dim) + ')'

class GGNN(nn.Module):
    def __init__(self, state_dim, annotation_dim, n_nodes, n_edge_types,
            n_steps):
        super().__init__()

        self.state_dim = state_dim
        self.annotation_dim = annotation_dim
        self.n_edge_types = n_edge_types
        self.n_nodes = n_nodes
        self.n_steps = n_steps

        for i in range(self.n_edge_types):
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)

            self.add_module('in_%i' % i, in_fc)
            self.add_module('out_%i' % i, out_fc)

        self.prop_model = PropModel(self.state_dim, self.n_nodes,
                self.n_edge_types)

        self.attn_layer = AttentionLayer(self.state_dim, self.state_dim)
        self.out = nn.Sequential(
                nn.Linear(self.state_dim + self.annotation_dim,
                    self.state_dim),
                nn.Tanh(),
                # nn.Linear(self.state_dim, 1)
            )

        # Initialize nodes
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

        self.in_fcs = AttrProxy(self, 'in_')
        self.out_fcs = AttrProxy(self, 'out_')


    def forward(self, init_hidden_state, annotation, adj_matrix, n_masked):
        hidden_state = init_hidden_state    # hidden_state: [batch_size, n_nodes, state_dim]

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_fc = self.in_fcs[i]
                out_fc = self.out_fcs[i]

                # Apply equation 8 from section 4 to infer the intermediate
                # annotations.
                in_states.append(in_fc(hidden_state))
                out_states.append(out_fc(hidden_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous() # in_states: [batch_size, n_edge_types, n_nodes, state_dim]
            in_states = in_states.view(-1, self.n_nodes * self.n_edge_types,
                    self.state_dim)

            out_states = torch.stack(out_states).transpose(0, 1).contiguous()   # out_states: [batch_size, n_edge_types, n_nodes, state_dim]
            out_states = out_states.view(-1, self.n_nodes * self.n_edge_types,
                    self.state_dim)

            hidden_state = self.prop_model(in_states, out_states, hidden_state, adj_matrix)
        
        in_adj_matrix, out_adj_matrix = torch.chunk(adj_matrix, 2, dim=2)
        n_edge_adj_matrixs = torch.chunk(in_adj_matrix, self.n_edge_types, dim=2)
        m_attn_states = []
        for etype in range(self.n_edge_types):
            attn_state = self.attn_layer(hidden_state, n_edge_adj_matrixs[etype])
            m_attn_states.append(attn_state)
        m_attn_states = torch.stack(m_attn_states, dim=0).transpose(0, 1)
        out_state = torch.mean(m_attn_states, dim=1)
        # Equation 7 of section 3.3
        output = self.out(torch.cat((out_state, annotation), 2))
        # print("output:", output.shape)
        p_output = output[:, 0]
        if n_masked:
            p_output = output[:, 1:n_masked+1]
            # print("n_masked >= 1:", p_output.shape)
        else:
            p_output = output[:, 0:1]
            # print("n_masked < 1:", p_output.shape)
        # output = output.sum(2)

        return p_output,output
