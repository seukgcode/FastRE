from paddle import fluid
import numpy as np
from .layers import Conv1D, GatedDilatedResidualConv1D, MultiHeadSelfAttention
from .utils import pos, gather


class SModel(fluid.dygraph.Layer):
    def __init__(self, embedding_size: int, hidden_size: int, max_len: int, num_subs: int):
        super(SModel, self).__init__(None)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.num_subs = num_subs

        self.position_embedding = fluid.dygraph.Embedding(size=[self.max_len, self.hidden_size])
        self.linear = fluid.dygraph.Linear(self.embedding_size, self.hidden_size)
        self.gdr1 = GatedDilatedResidualConv1D(self.hidden_size, 1)
        self.gdr2 = GatedDilatedResidualConv1D(self.hidden_size, 2)
        self.gdr3 = GatedDilatedResidualConv1D(self.hidden_size, 4)
        self.gdr4 = GatedDilatedResidualConv1D(self.hidden_size, 1)
        self.gdr5 = GatedDilatedResidualConv1D(self.hidden_size, 1)
        self.gdr6 = GatedDilatedResidualConv1D(self.hidden_size, 1)
        self.attention = MultiHeadSelfAttention(self.hidden_size, 8, 16)
        self.conv1d = Conv1D(input_dim=2 * self.hidden_size, output_dim=self.hidden_size, kernel_size=3, act="relu")
        self.sub_heads = fluid.dygraph.Linear(self.hidden_size, self.num_subs + 1)
        self.sub_tails = fluid.dygraph.Linear(self.hidden_size, self.num_subs + 1)

    def forward(self, token_ids, sent_vec):
        mask = fluid.layers.unsqueeze(token_ids, [2])
        mask = fluid.layers.cast((mask > fluid.dygraph.to_variable(np.array([0]))), "float32")
        position = pos(token_ids)
        position_embeddings = self.position_embedding(position)
        sent_vec = self.linear(sent_vec)
        features = fluid.layers.elementwise_add(sent_vec, position_embeddings)
        features = fluid.layers.dropout(features, dropout_prob=0.25) * mask
        features = self.gdr1(features, mask)
        features = self.gdr2(features, mask)
        features = self.gdr3(features, mask)
        features = self.gdr4(features, mask)
        features = self.gdr5(features, mask)
        features = self.gdr6(features, mask)
        attention_features = self.attention(features, features, features, mask)
        sub_features = fluid.layers.concat([features, attention_features], axis=-1)
        sub_features = self.conv1d(sub_features)
        pred_sub_heads = self.sub_heads(sub_features)
        pred_sub_tails = self.sub_tails(sub_features)
        return pred_sub_heads, pred_sub_tails, features, mask


class OPModel(fluid.dygraph.Layer):
    def __init__(self, hidden_size: int, type_size: int, max_len: int, num_relations: int, num_subs: int,
                 num_objs: int = None):
        super(OPModel, self).__init__(None)
        self.hidden_size = hidden_size
        self.type_size = type_size
        self.max_len = max_len
        self.num_relations = num_relations
        self.num_subs = num_subs
        self.num_objs = num_objs

        self.attention = MultiHeadSelfAttention(self.hidden_size, 8, 16)
        self.sub_head_embedding = fluid.dygraph.Embedding(size=[self.max_len, self.hidden_size])
        self.sub_tail_embedding = fluid.dygraph.Embedding(size=[self.max_len, self.hidden_size])
        self.sub_type_embedding = fluid.dygraph.Embedding(size=(self.num_subs, self.type_size))
        self.conv1d = Conv1D(input_dim=4 * self.hidden_size + self.type_size, output_dim=self.hidden_size,
                             kernel_size=3, act="relu")
        self.obj_heads = fluid.dygraph.Linear(self.hidden_size, self.num_relations + 1)
        self.obj_tails = fluid.dygraph.Linear(self.hidden_size, self.num_relations + 1)

    def forward(self, features, sub_id, sub_head, sub_tail, mask):
        attention_features = self.attention(features, features, features, mask)
        head_feature = gather(features, sub_head)
        tail_feature = gather(features, sub_tail)
        sub_feature = fluid.layers.concat([head_feature, tail_feature], -1)
        head_pos = self.sub_head_embedding(pos([features, sub_head]))
        tail_pos = self.sub_tail_embedding(pos([features, sub_tail]))
        pos_feature = fluid.layers.concat([head_pos, tail_pos], -1)
        sub_feature = fluid.layers.unsqueeze(sub_feature, 1) + pos_feature
        type_embedding = self.sub_type_embedding(sub_id)
        type_embedding = fluid.layers.expand(type_embedding, [1, features.shape[1], 1])
        obj_features = fluid.layers.concat([features, attention_features, sub_feature, type_embedding], axis=-1)
        obj_features = self.conv1d(obj_features)
        pred_obj_heads = self.obj_heads(obj_features)
        pred_obj_tails = self.obj_tails(obj_features)
        return pred_obj_heads, pred_obj_tails

