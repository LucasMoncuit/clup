from operator import itemgetter

import argparse
import sys

from gat.main import read_graphs
from split import completeSplit

#In AllenNLP we use type annotations for just about everything.
from typing import Iterator, List, Dict

#AllenNLP is built on top of PyTorch, so we use its code freely.
import torch
import torch.optim as optim
import numpy as np

#In AllenNLP we represent each training example as an Instance containing Fields of various types.
#Here each example will have a TextField containing the sentence, and a SequenceLabelField containing the corresponding part-of-speech tags.
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

#Typically to solve a problem like this using AllenNLP, you'll have to implement two classes.
#The first is a DatasetReader, which contains the logic for reading a file of data and producing a stream of Instances.
from allennlp.data.dataset_readers import DatasetReader

#Frequently we'll want to load datasets or models from URLs.
#The cached_path helper downloads such files, caches them locally, and returns the local path. It also accepts local file paths (which it just returns as-is).
from allennlp.common.file_utils import cached_path

#There are various ways to represent a word as one or more indices. For example, you might maintain a vocabulary of unique words and give each word a corresponding id.
#Or you might have one id per character in the word and represent each word as a sequence of ids. AllenNLP uses a has a TokenIndexer abstraction for this representation.
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

#Whereas a TokenIndexer represents a rule for how to turn a token into indices, a Vocabulary contains the corresponding mappings from strings to integers.
#For example, your token indexer might specify to represent a token as a sequence of character ids, in which case the Vocabulary would contain the mapping {character -> id}.
#In this particular example we use a SingleIdTokenIndexer that assigns each token a unique id, and so the Vocabulary will just contain a mapping {token -> id} (as well as the reverse mapping).
from allennlp.data.vocabulary import Vocabulary

#Besides DatasetReader, the other class you'll typically need to implement is Model,
#which is a PyTorch Module that takes tensor inputs and produces a dict of tensor outputs
#(including the training loss you want to optimize).
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
#As mentioned above, our model will consist of an embedding layer, followed by a LSTM, then by a feedforward layer.

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)

class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)
    def _read(self, file_path: str) -> Iterator[Instance]:
        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description="MRP Graph Toolkit");
            parser.add_argument("--normalize", action="store_true");
            parser.add_argument("--full", action="store_true");
            parser.add_argument("--reify", action="store_true");
            parser.add_argument("--format");
            parser.add_argument("input", nargs="?",
                                type=argparse.FileType("r"), default=sys.stdin);
            parser.add_argument("output", nargs="?",
                                type=argparse.FileType("w"), default=sys.stdout);
            arguments = parser.parse_args();
            graphs = read_graphs(arguments.input, format='mrp',
                             full=arguments.full, normalize=arguments.normalize,
                             reify=arguments.reify);
            for data in graphs:
                subgraphs = data.encode()
                tops = subgraphs['tops']
                nodes = subgraphs['nodes']
                all_edges = subgraphs['edges']
                input = subgraphs['input'].split()
                sentence = tuple(completeSplit(input))
                nodeTag = tuple([i for i in range(len(nodes))])
                sources = len(nodes)*[0]
                targets = len(nodes)*[0]
                for i in range(len(sentence),len(nodes)):
                    targets[i] = []
                for edge in all_edges:
                    for i in range(len(nodes)):
                        if edge['target'] == i:
                            sources[i] = [edge['source'],edge['label']]
                        if 0 in sources:
                            sources[sources.index(0)] = []
                        if edge['source'] == i:
                            j = edge['source']
                            targets[j] += [[edge['target'],edge['label']]]
                        for i in range(len(sentence), len(nodes)):
                            L = targets[i]
                            L.sort(key=itemgetter(0,1))
                            targets[i] = L
                for element in targets:
                    if isinstance(element, list):
                        if not element:
                            targets[targets.index(element)] = (None, None)
                        else:
                            for subelement in element:
                                element[element.index(subelement)] = tuple(subelement)
                        targets[targets.index(element)] = tuple(element)
                    else:
                        targets[targets.index(element)] = (None,None)
                for element in sources:
                    if not element:
                        sources[sources.index(element)] = (None, None)
                    else:
                        sources[sources.index(element)] = tuple(element)
                topTag = tuple(tops)
                tags = [(sources[i],targets[i]) for i in range(len(targets))]
                tags = tuple(tags)
                print(tags)

            '''yield self.text_to_instance([Token(word) for word in sentence], tags)'''
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

reader = PosDatasetReader()
train_dataset = reader.read('/home/lucas/PycharmProjects/clup/ucca/mrp/2019/training/ucca/ewt.mrp')
validation_dataset = reader.read('/home/lucas/PycharmProjects/clup/ucca/mrp/2019/training/ucca/wsj.mrp')
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmTagger(word_embeddings, lstm, vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000,
                  cuda_device=cuda_device)
trainer.train()
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
# Here's how to save the model.
with open("/home/lucas/PycharmProjects/clup/ucca/mrp/2019/models/model.th", 'wb') as f:
    torch.save(model.state_dict(), f)
vocab.save_to_files("/home/lucas/PycharmProjects/clup/ucca/mrp/2019/models/vocabulary")
# And here's how to reload the model.
vocab2 = Vocabulary.from_files("/home/lucas/PycharmProjects/clup/ucca/mrp/2019/models/vocabulary")
model2 = LstmTagger(word_embeddings, lstm, vocab2)
with open("/home/lucas/PycharmProjects/clup/ucca/mrp/2019/models/model.th", 'rb') as f:
    model2.load_state_dict(torch.load(f))
if cuda_device > -1:
    model2.cuda(cuda_device)
predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
np.testing.assert_array_almost_equal(tag_logits2, tag_logits)