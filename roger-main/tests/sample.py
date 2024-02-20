import os

import pytest


@pytest.fixture()
def openai_keys():
    openai_api_key = os.getenv("TEST_OPENAI_API_KEY", "")
    openai_org_key = os.getenv("TEST_OPENAI_ORG_KEY", "")

    assert openai_api_key != ""
    assert openai_org_key != ""

    return openai_api_key, openai_org_key


SHORT_TEXT = """We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from
Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from
unlabeled text by jointly conditioning on both
left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer
to create state-of-the-art models for a wide
range of tasks, such as question answering and
language inference, without substantial taskspecific architecture modifications.
BERT is conceptually simple and empirically
powerful. It obtains new state-of-the-art results on eleven natural language processing
tasks, including pushing the GLUE score to
80.5% (7.7% point absolute improvement),
MultiNLI accuracy to 86.7% (4.6% absolute
improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1
(5.1 point absolute improvement).
"""

LONG_TEXT = """Language model pre-training has been shown to
be effective for improving many natural language
processing tasks (Dai and Le, 2015; Peters et al.,
2018a; Radford et al., 2018; Howard and Ruder,
2018). These include sentence-level tasks such as
natural language inference (Bowman et al., 2015;
Williams et al., 2018) and paraphrasing (Dolan
and Brockett, 2005), which aim to predict the relationships between sentences by analyzing them
holistically, as well as token-level tasks such as
named entity recognition and question answering,
where models are required to produce fine-grained
output at the token level (Tjong Kim Sang and
De Meulder, 2003; Rajpurkar et al., 2016).
There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The
feature-based approach, such as ELMo (Peters
et al., 2018a), uses task-specific architectures that
include the pre-trained representations as additional features. The fine-tuning approach, such as
the Generative Pre-trained Transformer (OpenAI
GPT) (Radford et al., 2018), introduces minimal
task-specific parameters, and is trained on the
downstream tasks by simply fine-tuning all pretrained parameters. The two approaches share the
same objective function during pre-training, where
they use unidirectional language models to learn
general language representations.
We argue that current techniques restrict the
power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are
unidirectional, and this limits the choice of architectures that can be used during pre-training. For
example, in OpenAI GPT, the authors use a left-toright architecture, where every token can only attend to previous tokens in the self-attention layers
of the Transformer (Vaswani et al., 2017). Such restrictions are sub-optimal for sentence-level tasks,
and could be very harmful when applying finetuning based approaches to token-level tasks such
as question answering, where it is crucial to incorporate context from both directions.
In this paper, we improve the fine-tuning based
approaches by proposing BERT: Bidirectional
Encoder Representations from Transformers.
BERT alleviates the previously mentioned unidirectionality constraint by using a “masked language model” (MLM) pre-training objective, inspired by the Cloze task (Taylor, 1953). The
masked language model randomly masks some of
the tokens from the input, and the objective is to
predict the original vocabulary id of the masked
word based only on its context. Unlike left-toright language model pre-training, the MLM objective enables the representation to fuse the left
and the right context, which allows us to pretrain a deep bidirectional Transformer. In addition to the masked language model, we also use
a “next sentence prediction” task that jointly pretrains text-pair representations. The contributions
of our paper are as follows:
• We demonstrate the importance of bidirectional
pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT
uses masked language models to enable pretrained deep bidirectional representations. This
is also in contrast to Peters et al. (2018a), which
uses a shallow concatenation of independently
trained left-to-right and right-to-left LMs.
• We show that pre-trained representations reduce
the need for many heavily-engineered taskspecific architectures. BERT is the first finetuning based representation model that achieves
state-of-the-art performance on a large suite
of sentence-level and token-level tasks, outperforming many task-specific architectures.
• BERT advances the state of the art for eleven
NLP tasks. The code and pre-trained models are available at https://github.com/
google-research/bert.
2 Related Work
There is a long history of pre-training general language representations, and we briefly review the
most widely-used approaches in this section.
2.1 Unsupervised Feature-based Approaches
Learning widely applicable representations of
words has been an active area of research for
decades, including non-neural (Brown et al., 1992;
Ando and Zhang, 2005; Blitzer et al., 2006) and
neural (Mikolov et al., 2013; Pennington et al.,
2014) methods. Pre-trained word embeddings
are an integral part of modern NLP systems, offering significant improvements over embeddings
learned from scratch (Turian et al., 2010). To pretrain word embedding vectors, left-to-right language modeling objectives have been used (Mnih
and Hinton, 2009), as well as objectives to discriminate correct from incorrect words in left and
right context (Mikolov et al., 2013).
These approaches have been generalized to
coarser granularities, such as sentence embeddings (Kiros et al., 2015; Logeswaran and Lee,
2018) or paragraph embeddings (Le and Mikolov,
2014). To train sentence representations, prior
work has used objectives to rank candidate next
sentences (Jernite et al., 2017; Logeswaran and
Lee, 2018), left-to-right generation of next sentence words given a representation of the previous
sentence (Kiros et al., 2015), or denoising autoencoder derived objectives (Hill et al., 2016).
ELMo and its predecessor (Peters et al., 2017,
2018a) generalize traditional word embedding research along a different dimension. They extract
context-sensitive features from a left-to-right and a
right-to-left language model. The contextual representation of each token is the concatenation of
the left-to-right and right-to-left representations.
When integrating contextual word embeddings
with existing task-specific architectures, ELMo
advances the state of the art for several major NLP
benchmarks (Peters et al., 2018a) including question answering (Rajpurkar et al., 2016), sentiment
analysis (Socher et al., 2013), and named entity
recognition (Tjong Kim Sang and De Meulder,
2003). Melamud et al. (2016) proposed learning
contextual representations through a task to predict a single word from both left and right context
using LSTMs. Similar to ELMo, their model is
feature-based and not deeply bidirectional. Fedus
et al. (2018) shows that the cloze task can be used
to improve the robustness of text generation models."""
