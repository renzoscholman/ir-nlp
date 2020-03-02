import stanfordnlp

class NLPRootDist:
    # Can be initiated with other models location and other treebanks (en_ewt, en_gum, en_lines)
    # Tested with Stanford CoreNLP models for English: en_ewt, en_gum and en_lines, all version 0.2.0.
    # Default model is en_ewt
    def __init__(self, model_location='./corenlp_models', treebank='en_ewt'):
        self.nlp = stanfordnlp.Pipeline(lang="en", treebank=treebank, models_dir=model_location)

    def parse(self, sentence):
        return self.nlp(sentence)
