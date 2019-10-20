# NLP - Tutorial
Repository to show how NLP can tacke real problem. Including the source code, dataset, state-of-the art in NLP

## Data Augmentation
*   [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28)
*   [Data Augmentation library for Text](https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
*   [Does your NLP model able to prevent adversarial attack?](https://medium.com/hackernoon/does-your-nlp-model-able-to-prevent-adversarial-attack-45b5ab75129c)
*   [How does Data Noising Help to Improve your NLP Model?](https://medium.com/towards-artificial-intelligence/how-does-data-noising-help-to-improve-your-nlp-model-480619f9fb10)
*   [Data Augmentation library for Speech Recognition](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78)
*   [Data Augmentation library for Audio](https://towardsdatascience.com/data-augmentation-for-audio-76912b01fdf6)
*   [Unsupervied Data Augmentation](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143)
*   [Adversarial Attacks in Textual Deep Neural Networks](https://medium.com/towards-artificial-intelligence/adversarial-attacks-in-textual-deep-neural-networks-245dc90029df)


## Text Preprocessing
| Section | Sub-Section | Description | Story |
| --- | --- | --- | --- |
| Tokenization | Subword Tokenization |  | [Medium](https://towardsdatascience.com/how-subword-helps-on-your-nlp-model-83dd1b836f46) |
| Tokenization | Word Tokenization |  | [Medium](https://medium.com/@makcedward/nlp-pipeline-word-tokenization-part-1-4b2b547e6a3) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-word_tokenization.ipynb) |
| Tokenization | Sentence Tokenization |  | [Medium](https://medium.com/@makcedward/nlp-pipeline-sentence-tokenization-part-6-86ed55b185e6) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-sentence_tokenization.ipynb) |
| Part of Speech | | | [Medium](https://medium.com/@makcedward/nlp-pipeline-part-of-speech-part-2-b683c90e327d) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-part_of_speech.ipynb) |
| Lemmatization | | | [Medium](https://medium.com/@makcedward/nlp-pipeline-lemmatization-part-3-4bfd7304957) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp_lemmatization.ipynb) |
| Stemming | | | [Medium](https://medium.com/@makcedward/nlp-pipeline-stemming-part-4-b60a319fd52) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-stemming.ipynb) |
| Stop Words | | | [Medium](https://medium.com/@makcedward/nlp-pipeline-stop-words-part-5-d6770df8a936) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-stop_words.ipynb) |
| Phrase Word Recognition | |  |  |
| Spell Checking | Lexicon-based | Peter Norvig algorithm | [Medium](https://towardsdatascience.com/correcting-your-spelling-error-with-4-operations-50bcfd519bb8) [Github](https://github.com/makcedward/nlp/blob/master/sample/util/nlp-util-spell_corrector.ipynb) |
| | Lexicon-based | Symspell | [Medium](https://towardsdatascience.com/essential-text-correction-process-for-nlp-tasks-f731a025fcc3) [Github](https://github.com/makcedward/nlp/blob/master/sample/util/nlp-util-symspell.ipynb) |
| | Machine Translation | Statistical Machine Translation | [Medium](https://towardsdatascience.com/correcting-text-input-by-machine-translation-and-classification-fa9d82087de1) |
| | Machine Translation | Attention | [Medium](https://towardsdatascience.com/fix-your-text-thought-attention-before-nlp-tasks-7dc074b9744f) |
| String Matching | Fuzzywuzzy | | [Medium](https://towardsdatascience.com/how-fuzzy-matching-improve-your-nlp-model-bc617385ad6b) [Github](https://github.com/makcedward/nlp/blob/master/sample/preprocessing/nlp-preprocessing-string_matching-fuzzywuzzy.ipynb) |

## Text Representation
| Section | Sub-Section | Research Lab | Story | Source |
| --- | --- | --- | --- | --- |
| Traditional Method | Bag-of-words (BoW) |  | [Medium](https://towardsdatascience.com/3-basic-approaches-in-bag-of-words-which-are-better-than-word-embeddings-c2cbc7398016) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-bag_of_words.ipynb) |  |
|  | Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA) |  | [Medium](https://towardsdatascience.com/2-latent-methods-for-dimension-reduction-and-topic-modeling-20ff6d7d547) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-lsa_lda.ipynb) |  |
| Character Level | Character Embedding | NYU | [Medium](https://medium.com/@makcedward/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-character_embedding.ipynb) | [Paper](https://arxiv.org/pdf/1502.01710v5.pdf) |
| Word Level | Negative Sampling and Hierarchical Softmax |  | [Medium](https://towardsdatascience.com/how-negative-sampling-work-on-word2vec-7bf8d545b116) |  |
|  | Word2Vec, GloVe, fastText |  | [Medium](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-word_embedding.ipynb) |  |
|  | Contextualized Word Vectors (CoVe) | Salesforce | [Medium](https://towardsdatascience.com/replacing-your-word-embeddings-by-contextualized-word-vectors-9508877ad65d) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-embeddings-word-cove.ipynb) | [Paper](http://papers.nips.cc/paper/7209-learned-in-translation-contextualized-word-vectors.pdf) [Code](https://github.com/salesforce/cove) |
|  | Misspelling Oblivious (word) Embeddings | Facebook | [Medium](https://medium.com/towards-artificial-intelligence/new-model-for-word-embeddings-which-are-resilient-to-misspellings-moe-9ecfd3ab473e) | [Paper](https://arxiv.org/pdf/1905.09755.pdf) |
|  | Embeddings from Language Models (ELMo) | AI2 | [Medium](https://towardsdatascience.com/elmo-helps-to-further-improve-your-word-embeddings-c6ed2c9df95f) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-embeddings-sentence-elmo.ipynb) | [Paper](https://arxiv.org/pdf/1802.05365.pdf) [Code](https://github.com/allenai/allennlp/) |
|  | Contextual String Embeddings | Zalando Research | [Medium](https://towardsdatascience.com/contextual-embeddings-for-nlp-sequence-labeling-9a92ba5a6cf0) | [Paper](http://aclweb.org/anthology/C18-1139) [Code](https://github.com/zalandoresearch/flair)| 
| Sentence Level | Skip-thoughts |  | [Medium](https://towardsdatascience.com/transforming-text-to-sentence-embeddings-layer-via-some-thoughts-b77bed60822c) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-skip_thoughts.ipynb) | [Paper](https://arxiv.org/pdf/1506.06726) [Code](https://github.com/ryankiros/skip-thoughts) |
|  | InferSent |  | [Medium](https://towardsdatascience.com/learning-sentence-embeddings-by-natural-language-inference-a50b4661a0b8) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-embeddings-sentence-infersent.ipynb) | [Paper](https://arxiv.org/pdf/1705.02364) [Code](https://github.com/facebookresearch/InferSent) |
|  | Quick-Thoughts | Google | [Medium](https://towardsdatascience.com/building-sentence-embeddings-via-quick-thoughts-945484cae273) | [Paper](https://arxiv.org/pdf/1803.02893.pdf) [Code](https://github.com/lajanugen/S2V) |
|  | General Purpose Sentence (GenSen) |  | [Medium](https://towardsdatascience.com/learning-generic-sentence-representation-by-various-nlp-tasks-df39ce4e81d7) | [Paper](https://arxiv.org/pdf/1804.00079.pdf) [Code](https://github.com/Maluuba/gensen) |
|  | Bidirectional Encoder Representations from Transformers (BERT) | Google | [Medium](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb) | [Paper(2019)](https://arxiv.org/pdf/1810.04805) [Code](https://github.com/google-research/bert)| 
|  | Generative Pre-Training (GPT) | OpenAI | [Medium](https://towardsdatascience.com/combining-supervised-learning-and-unsupervised-learning-to-improve-word-vectors-d4dea84ec36b) | [Paper(2019)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) [Code](https://github.com/openai/finetune-transformer-lm)| 
|  | Self-Governing Neural Networks (SGNN) | Google | [Medium](https://towardsdatascience.com/embeddings-free-deep-learning-nlp-model-ce067c7a7c93) | [Paper](https://aclweb.org/anthology/D18-1105) | 
|  | Multi-Task Deep Neural Networks (MT-DNN) | Microsoft | [Medium](https://towardsdatascience.com/when-multi-task-learning-meet-with-bert-d1c49cc40a0c) | [Paper(2019)](https://arxiv.org/pdf/1901.11504.pdf) | 
|  | Generative Pre-Training-2 (GPT-2) | OpenAI | [Medium](https://towardsdatascience.com/too-powerful-nlp-model-generative-pre-training-2-4cc6afb6655) | [Paper(2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [Code](https://github.com/openai/gpt-2)| 
|  | Universal Language Model Fine-tuning (ULMFiT) | OpenAI | [Medium](https://towardsdatascience.com/multi-task-learning-in-language-model-for-text-classification-c3acc1fedd89) | [Paper](https://arxiv.org/pdf/1801.06146.pdf) [Code](https://github.com/fastai/fastai)| 
|  | BERT in Science Domain |  | [Medium](https://towardsdatascience.com/how-to-apply-bert-in-scientific-domain-2d9db0480bd9) | [Paper(2019)](https://arxiv.org/pdf/1903.10676.pdf) [Paper(2019)](https://arxiv.org/pdf/1901.08746.pdf)| 
|  | BERT in Clinical Domain | NYU/PU | [Medium](https://towardsdatascience.com/how-do-they-apply-bert-in-the-clinical-domain-49113a51be50) | [Paper(2019)](https://arxiv.org/pdf/1904.03323.pdf) [Paper(2019)](https://arxiv.org/pdf/1904.05342.pdf)| 
|  | RoBERTa | UW/Facebook | [Medium](https://medium.com/towards-artificial-intelligence/a-robustly-optimized-bert-pretraining-approach-f6b6e537e6a6) | [Paper(2019)](https://arxiv.org/pdf/1904.03323.pdf) [Paper](https://arxiv.org/pdf/1907.11692.pdf)| 
|  | Unified Language Model for NLP and NLU (UNILM) | Microsoft | [Medium](https://medium.com/towards-artificial-intelligence/unified-language-model-pre-training-for-natural-language-understanding-and-generation-f87dc226aa2) | [Paper(2019)](https://arxiv.org/pdf/1905.03197.pdf)| 
|  | Cross-lingual Language Model (XLMs) | Facebook | [Medium](https://medium.com/towards-artificial-intelligence/cross-lingual-language-model-56a65dba9358) | [Paper(2019)](https://arxiv.org/pdf/1901.07291.pdf)| 
|  | Transformer-XL | CMU/Google | [Medium](https://medium.com/towards-artificial-intelligence/address-limitation-of-rnn-in-nlp-problems-by-using-transformer-xl-866d7ce1c8f4) | [Paper(2019)](https://arxiv.org/pdf/1901.02860.pdf)| 
|  | XLNet | CMU/Google | [Medium](https://medium.com/dataseries/why-does-xlnet-outperform-bert-da98a8503d5b) | [Paper(2019)](https://arxiv.org/pdf/1906.08237.pdf)| 
|  | CTRL | Salesforce | [Medium](https://medium.com/dataseries/a-controllable-framework-for-text-generation-8be9e1f2c5db) | [Paper(2019)](https://arxiv.org/pdf/1909.05858.pdf)| 
| Document Level | lda2vec |  | [Medium](https://towardsdatascience.com/combing-lda-and-word-embeddings-for-topic-modeling-fe4a1315a5b4) | [Paper](https://arxiv.org/pdf/1605.02019.pdf) |
|  | doc2vec | Google | [Medium](https://towardsdatascience.com/understand-how-to-transfer-your-paragraph-to-vector-by-doc2vec-1e225ccf102) [Github](https://github.com/makcedward/nlp/blob/master/sample/embeddings/nlp-embeddings-document-doc2vec.ipynb) | [Paper](https://arxiv.org/pdf/1405.4053.pdf) |

## NLP Problem 
| Section | Sub-Section | Description | Research Lab | Story | Paper & Code |
| --- | --- | --- | --- | --- | --- |
| Named Entity Recognition (NER) | Pattern-based Recognition | | | [Medium](https://towardsdatascience.com/pattern-based-recognition-did-help-in-nlp-5c54b4e7a962)  |  |
| | Lexicon-based Recognition | | | [Medium](https://towardsdatascience.com/step-out-from-regular-expression-for-feature-engineering-134e594f542c) |  |
| | spaCy Pre-trained NER | | | [Medium](https://medium.com/@makcedward/named-entity-recognition-3fad3f53c91e) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-named_entity_recognition.ipynb) |  |
| Optical Character Recognition (OCR) | Printed Text | Google Cloud Vision API | Google | [Medium](https://towardsdatascience.com/secret-of-google-web-based-ocr-service-fe30eecedd01) | [Paper](https://das2018.cvl.tuwien.ac.at/media/filer_public/85/fd/85fd4698-040f-45f4-8fcc-56d66533b82d/das2018_short_papers.pdf) |
| | Handwriting | LSTM | Google | [Medium](https://towardsdatascience.com/lstm-based-handwriting-recognition-by-google-eb99663ca6de) | [Paper](https://arxiv.org/pdf/1902.10525.pdf) | 
| Text Summarization | Extractive Approach | | | [Medium](https://medium.com/@makcedward/text-summarization-extractive-approach-567fe4b85c23) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-text_summarization_extractive.ipynb) | |
| | Abstractive Approach |  |  | [Medium](https://medium.com/dataseries/summarize-document-by-combing-extractive-and-abstractive-steps-40295310526) | 
| Emotion Recognition | Audio, Text, Visual | 3 Multimodals for Emotion Recognition |  | [Medium](https://becominghuman.ai/multimodal-for-emotion-recognition-21df267fddc4) |

## Acoustic Problem
| Section | Sub-Section | Description | Research Lab | Story | Paper & Code |
| --- | --- | --- | --- | --- | --- |
| Feature Representation | Unsupervised Learning| Introduction to Audio Feature Learning | |  [Medium](https://medium.com/hackernoon/how-can-you-apply-unsupervised-learning-on-audio-data-be95153c5860) | [Paper 1](https://ai.stanford.edu/~ang/papers/nips09-AudioConvolutionalDBN.pdf) [Paper 2](https://arxiv.org/pdf/1607.03681.pdf) [Paper 3](https://arxiv.org/pdf/1712.03835.pdf)
| Feature Representation | Unsupervised Learning| Speech2Vec and Sentence Level Embeddings | |  [Medium](https://medium.com/towards-artificial-intelligence/two-ways-to-learn-audio-embeddings-9dfcaab10ba6) | [Paper 1](https://arxiv.org/pdf/1803.08976.pdf) [Paper 2](https://arxiv.org/pdf/1902.07817.pdf)
| Speech-to-text | | Introduction to Speeh-to-text | |  [Medium](https://becominghuman.ai/how-does-your-assistant-device-work-based-on-text-to-speech-technology-5f31e56eae7e) |

## Text Distance Measurement
| Section | Sub-Section | Description | Research Lab | Story | Paper & Code |
| --- | --- | --- | --- | --- | --- |
| Euclidean Distance, Cosine Similarity and Jaccard Similarity |  |  |  | [Medium](https://towardsdatascience.com/3-basic-distance-measurement-in-text-mining-5852becff1d7) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-3_basic_distance_measurement_in_text_mining.ipynb) |  |
| Edit Distance | Levenshtein Distance |  |  | [Medium](https://towardsdatascience.com/measure-distance-between-2-words-by-simple-calculation-a97cf4993305) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-distance-edit_distance.ipynb) |  |
| Word Moving Distance (WMD) |  |  |  | [Medium](https://towardsdatascience.com/word-distance-between-word-embeddings-cc3e9cf1d632) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-word_mover_distance.ipynb) |
| Supervised Word Moving Distance (S-WMD) |  |  |  | [Medium](https://towardsdatascience.com/word-distance-between-word-embeddings-with-weight-bf02869c50e1)|
| Manhattan LSTM |  |  |  | [Medium](https://towardsdatascience.com/text-matching-with-deep-learning-e6aa05333399) | [Paper](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) |

## Model Interpretation
| Section | Sub-Section | Description | Research Lab | Story | Paper & Code |
| --- | --- | --- | --- | --- | --- |
| ELI5, LIME and Skater |  |  |  | [Medium](https://towardsdatascience.com/3-ways-to-interpretate-your-nlp-model-to-management-and-customer-5428bc07ce15) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-model_interpretation.ipynb) |
| SHapley Additive exPlanations (SHAP) |  |  |  | [Medium](https://towardsdatascience.com/interpreting-your-deep-learning-model-by-shap-e69be2b47893) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-model_interpretation_shap.ipynb) |
| Anchors |  |  |  | [Medium](https://towardsdatascience.com/anchor-your-model-interpretation-by-anchors-aa4ed7104032) [Github](https://github.com/makcedward/nlp/blob/master/sample/nlp-model_interpretation_anchor.ipynb) |

## Source Code 
| Section | Sub-Section | Description | Link |
| --- | --- | --- | --- |
| Spellcheck |  |  | [Github](https://github.com/norvig/pytudes) |
| InferSent |  |  | [Github](https://github.com/facebookresearch/InferSent) |
