from paddle import nn
from paddlenlp.transformers import ErnieModel, AutoModelForTokenClassification, ErniePretrainedModel, PretrainedModel
from paddlenlp.transformers.model_outputs import SequenceClassifierOutput
from typing import Optional, Tuple
from paddle import Tensor


class ErnieForClsAndSeq(nn.Layer):
    def __init__(self, model_name, num_labels, num_classes):
        super(ErnieForClsAndSeq, self).__init__()
        self.ernie_model = ErnieModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.ernie_model.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie_model.config["hidden_size"], num_classes)
        self.sequence = nn.Linear(self.ernie_model.config["hidden_size"], num_labels)

    def forward(
            self,
            input_ids: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        sequence_output, pooled_output = self.ernie_model(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        pooled_logits = self.classifier(pooled_output)

        sequence_output = self.dropout(sequence_output)
        sequence_logits = self.sequence(sequence_output)

        return pooled_logits, sequence_logits
