import stanfordnlp

class NLPRootDist:
    # Can be initiated with other models location and other treebanks (en_ewt, en_gum, en_lines)
    def __init__(self, model_location='./corenlp_models', treebank='en_ewt'):
        self.nlp = stanfordnlp.Pipeline(lang="en", treebank=treebank, models_dir=model_location)
        self.hedging_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]
        self.refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

    def build_dep_graph(self, deps):
        dep_graph = {}
        dep_graph_labels = {}

        for d in deps:
            head, rel, dep = d
            dep_graph.setdefault(int(head.index), set()).add(int(dep.index))
            dep_graph_labels[(int(head.index), int(dep.index))] = rel
        return dep_graph, dep_graph_labels

    def calc_depths(self, grph, n=0, d=0, depths=None):
        if depths is None:
            depths = {n: d}
        sx = grph.get(n)
        if sx:
            for s in sx:
                depths[s] = d + 1
                self.calc_depths(grph, s, d + 1, depths)
        return depths

    def parse(self, sentence, print_words=False):
        doc = self.nlp(sentence)
        min_hedge_depth = min_refute_depth = 100
        for i, sent in enumerate(doc.sentences):
            grph, grph_labels = self.build_dep_graph(sent.dependencies)
            grph_depths = self.calc_depths(grph)
            lemmas = list(enumerate([d.lemma.lower() for d in sent.words], start=1))
            h_depths = [h if lem in self.hedging_words else 0 for (h, lem) in lemmas]
            r_depths = [h if lem in self.refuting_words else 0 for (h, lem) in lemmas]

            hedge_depths = [grph_depths[d] for d in h_depths if d > 0]
            refute_depths = [grph_depths[d] for d in r_depths if d > 0]

            hedge_depths.append(min_hedge_depth)
            refute_depths.append(min_refute_depth)

            min_hedge_depth = min(hedge_depths)
            min_refute_depth = min(refute_depths)
            if print_words:
                print(*[f'text: {word.text + " "}\tlemma: {word.lemma}\tupos: {word.upos}\txpos: {word.xpos} {word}' for word in sent.words], sep='\n')
        return min_hedge_depth, min_refute_depth