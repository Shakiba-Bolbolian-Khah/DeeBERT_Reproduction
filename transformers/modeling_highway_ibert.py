import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_ibert import IBertLayer, IBertPreTrainedModel, IBertClassificationHead,create_position_ids_from_input_ids
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear
from .modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

def entropy(x):
    # x: torch.Tensor, logits BEFORE softmax
    x = torch.softmax(x, dim=-1)               # softmax normalized prob distribution
    return -torch.sum(x*torch.log(x), dim=-1)  # entropy calculation on probs: -\sum(p \ln(p))


class IBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.embedding_bit = 8
        self.embedding_act_bit = 16
        self.act_bit = 8
        self.ln_input_bit = 22
        self.ln_output_bit = 32

        self.word_embeddings = QuantEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )
        self.token_type_embeddings = QuantEmbedding(
            config.type_vocab_size, config.hidden_size, weight_bit=self.embedding_bit, quant_mode=self.quant_mode
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = QuantEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )

        # Integer-only addition between embeddings
        self.embeddings_act1 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.embeddings_act2 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = IntLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds, inputs_embeds_scaling_factor = self.word_embeddings(input_ids)
        else:
            inputs_embeds_scaling_factor = None
        token_type_embeddings, token_type_embeddings_scaling_factor = self.token_type_embeddings(token_type_ids)

        embeddings, embeddings_scaling_factor = self.embeddings_act1(
            inputs_embeds,
            inputs_embeds_scaling_factor,
            identity=token_type_embeddings,
            identity_scaling_factor=token_type_embeddings_scaling_factor,
        )

        if self.position_embedding_type == "absolute":
            position_embeddings, position_embeddings_scaling_factor = self.position_embeddings(position_ids)
            embeddings, embeddings_scaling_factor = self.embeddings_act1(
                embeddings,
                embeddings_scaling_factor,
                identity=position_embeddings,
                identity_scaling_factor=position_embeddings_scaling_factor,
            )

        embeddings, embeddings_scaling_factor = self.LayerNorm(embeddings, embeddings_scaling_factor)
        embeddings = self.dropout(embeddings)
        embeddings, embeddings_scaling_factor = self.output_activation(embeddings, embeddings_scaling_factor)
        return embeddings, embeddings_scaling_factor

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class IBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quant_mode = config.quant_mode
        self.layer = nn.ModuleList([IBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.highway = nn.ModuleList([IBertHighway(config) for _ in range(config.num_hidden_layers)])

        self.early_exit_entropy = [-1 for _ in range(config.num_hidden_layers)]

    def set_early_exit_entropy(self, x):
        print(x)
        if (type(x) is float) or (type(x) is int):
            for i in range(len(self.early_exit_entropy)):
                self.early_exit_entropy[i] = x
        else:
            self.early_exit_entropy = x

    def init_highway_pooler(self, pooler):
        loaded_model = pooler.state_dict()
        for highway in self.highway:
            for name, param in highway.pooler.state_dict().items():
                param.copy_(loaded_model[name])

    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        all_hidden_states = () 
        all_self_attentions = () 
        all_highway_exits = ()

        all_cross_attentions = None  # `config.add_cross_attention` is not supported
        next_decoder_cache = None  # `config.use_cache` is not supported

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                raise NotImplementedError("gradient checkpointing is not currently supported")

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    hidden_states_scaling_factor,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            #######
            current_outputs = (hidden_states,)
            if output_hidden_states:
                current_outputs = current_outputs + (all_hidden_states,)
            if output_attentions:
                current_outputs = current_outputs + (all_self_attentions,)

            highway_exit = self.highway[i](current_outputs)
            # logits, pooled_output

            if not self.training:
                highway_logits = highway_exit[0]
                highway_entropy = entropy(highway_logits)
                highway_exit = highway_exit + (highway_entropy,)  # logits, hidden_states(?), entropy
                all_highway_exits = all_highway_exits + (highway_exit,)

                if highway_entropy < self.early_exit_entropy[i]:
                    # weight_func = lambda x: torch.exp(-3 * x) - 0.5**3
                    # weight_func = lambda x: 2 - torch.exp(x)
                    # weighted_logits = \
                    #     sum([weight_func(x[2]) * x[0] for x in all_highway_exits]) /\
                    #     sum([weight_func(x[2]) for x in all_highway_exits])
                    # new_output = (weighted_logits,) + current_outputs[1:] + (all_highway_exits,)
                    new_output = (highway_logits,) + current_outputs[1:] + (all_highway_exits,)
                    raise HighwayException(new_output, i+1)
            else:
                all_highway_exits = all_highway_exits + (highway_exit,)

        ########

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_self_attentions,)

        outputs = outputs + (all_highway_exits,)
        return outputs 
        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [
        #             hidden_states,
        #             next_decoder_cache,
        #             all_hidden_states,
        #             all_self_attentions,
        #             all_cross_attentions,
        #         ]
        #         if v is not None
        #     )
        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_decoder_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


class IBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class IBertModel(IBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.quant_mode = config.quant_mode

        self.embeddings = IBertEmbeddings(config)
        self.encoder = IBertEncoder(config)

        self.pooler = IBertPooler(config) #if add_pooling_layer else None

        self.init_weights()

######
    def init_highway_pooler(self):
        self.encoder.init_highway_pooler(self.pooler)
########

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = None#return_dict if return_dict is not None else None #self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        encoder_attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers


        embedding_output, embedding_output_scaling_factor = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            embedding_output_scaling_factor,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) #if self.pooler is not None else None

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs
        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )

class HighwayException(Exception):
    def __init__(self, message, exit_layer):
        self.message = message
        self.exit_layer = exit_layer  # start from 1!

class IBertHighway(nn.Module):
    r"""A module to provide a shortcut
    from
    the output of one non-final BertLayer in BertEncoder
    to
    cross-entropy computation in BertForSequenceClassification
    """
    def __init__(self, config):
        super(IBertHighway, self).__init__()
        self.pooler = IBertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs[0]
        pooler_output = self.pooler(pooler_input)
        # "return" pooler_output

        # IBertModel
        bmodel_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        # "return" bodel_output

        # Dropout and classification
        pooled_output = bmodel_output[1]

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, pooled_output


class IBertForSequenceClassification(IBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.ibert = IBertModel(config, add_pooling_layer=True)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_layer=-1, train_highway=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = None#return_dict if return_dict is not None else None #else self.config.use_return_dict
        exit_layer = self.num_layers
        try:
            outputs = self.ibert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            pooled_output = outputs[1]

            # pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            outputs = (logits,) + outputs[2:]

        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        if not self.training:
            original_entropy = entropy(logits)
            highway_entropy = []
            highway_logits_all = []


        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

####
            # work with highway exits
            highway_losses = []
            for highway_exit in outputs[-1]:
                highway_logits = highway_exit[0]
                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_entropy.append(highway_exit[2])
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    highway_loss = loss_fct(highway_logits.view(-1),
                                            labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    highway_loss = loss_fct(highway_logits.view(-1, self.num_labels),
                                            labels.view(-1))
                highway_losses.append(highway_loss)

            if train_highway:
                outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course
            else:
                outputs = (loss,) + outputs
        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) +\
                          (highway_logits_all[output_layer],) +\
                          outputs[2:]  ## use the highway of the last layer

        return outputs









































# from __future__ import (absolute_import, division, print_function,
#                         unicode_literals)

# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss, MSELoss

# from .modeling_ibert import IBertEmbeddings
# from .modeling_highway_bert import BertModel, BertPreTrainedModel, entropy, HighwayException
# from .configuration_ibert import IBertConfig

# IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "kssteven/ibert-roberta-base": "https://huggingface.co/kssteven/ibert-roberta-base/resolve/main/config.json",
#     "kssteven/ibert-roberta-large": "https://huggingface.co/kssteven/ibert-roberta-large/resolve/main/config.json",
#     "kssteven/ibert-roberta-large-mnli": "https://huggingface.co/kssteven/ibert-roberta-large-mnli/resolve/main/config.json",
# }


# class IBertModel(BertModel):
#     r"""
#     Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#         **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
#             Sequence of hidden-states at the output of the last layer of the model.
#         **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
#             Last layer hidden-state of the first token of the sequence (classification token)
#             further processed by a Linear layer and a Tanh activation function. The Linear
#             layer weights are trained from the next sentence prediction (classification)
#             objective during Bert pretraining. This output is usually *not* a good summary
#             of the semantic content of the input, you're often better with averaging or pooling
#             the sequence of hidden-states for the whole input sequence.
#         **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
#             list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
#             of shape ``(batch_size, sequence_length, hidden_size)``:
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         **attentions**: (`optional`, returned when ``config.output_attentions=True``)
#             list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#     Examples::

#         tokenizer = RobertaTokenizer.from_pretrained("kssteven/ibert-roberta-base")
#         model = IBertForSequenceClassification.from_pretrained("kssteven/ibert-roberta-base")
#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids)
#         last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

#     """
#     config_class = IBertConfig
#     pretrained_model_archive_map = IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
#     base_model_prefix = "kssteven/ibert-roberta"

#     def __init__(self, config):
#         super(IBertModel, self).__init__(config)

#         self.embeddings = IBertEmbeddings(config)
#         self.init_weights()

#     def get_input_embeddings(self):
#         return self.embeddings.word_embeddings

#     def set_input_embeddings(self, value):
#         self.embeddings.word_embeddings = value


# class IBertForSequenceClassification(BertPreTrainedModel):
#     r"""
#         **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in ``[0, ..., config.num_labels]``.
#             If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
#             If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

#     Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#         **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#             Classification (or regression if config.num_labels==1) loss.
#         **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
#             Classification (or regression if config.num_labels==1) scores (before SoftMax).
#         **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
#             list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
#             of shape ``(batch_size, sequence_length, hidden_size)``:
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         **attentions**: (`optional`, returned when ``config.output_attentions=True``)
#             list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.


#     """
#     config_class = IBertConfig
#     pretrained_model_archive_map = IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
#     base_model_prefix = "kssteven/ibert-roberta"

#     def __init__(self, config):
#         super(IBertForSequenceClassification, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.num_layers = config.num_hidden_layers

#         self.ibert = IBertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None,
#                 labels=None,
#                 output_layer=-1, train_highway=False):

#         exit_layer = self.num_layers
#         try:
#             outputs = self.ibert(input_ids,
#                                    attention_mask=attention_mask,
#                                    token_type_ids=token_type_ids,
#                                    position_ids=position_ids,
#                                    head_mask=head_mask,
#                                    inputs_embeds=inputs_embeds)

#             pooled_output = outputs[1]

#             pooled_output = self.dropout(pooled_output)
#             logits = self.classifier(pooled_output)
#             outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
#         except HighwayException as e:
#             outputs = e.message
#             exit_layer = e.exit_layer
#             logits = outputs[0]

#         if not self.training:
#             original_entropy = entropy(logits)
#             highway_entropy = []
#             highway_logits_all = []
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#             # work with highway exits
#             highway_losses = []
#             for highway_exit in outputs[-1]:
#                 highway_logits = highway_exit[0]
#                 if not self.training:
#                     highway_logits_all.append(highway_logits)
#                     highway_entropy.append(highway_exit[2])
#                 if self.num_labels == 1:
#                     #  We are doing regression
#                     loss_fct = MSELoss()
#                     highway_loss = loss_fct(highway_logits.view(-1),
#                                             labels.view(-1))
#                 else:
#                     loss_fct = CrossEntropyLoss()
#                     highway_loss = loss_fct(highway_logits.view(-1, self.num_labels),
#                                             labels.view(-1))
#                 highway_losses.append(highway_loss)

#             if train_highway:
#                 outputs = (sum(highway_losses[:-1]),) + outputs
#                 # exclude the final highway, of course
#             else:
#                 outputs = (loss,) + outputs
#         if not self.training:
#             outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
#             if output_layer >= 0:
#                 outputs = (outputs[0],) + \
#                           (highway_logits_all[output_layer],) + \
#                           outputs[2:]  ## use the highway of the last layer

#         return outputs  # (loss), logits, (hidden_states), (attentions), entropy
