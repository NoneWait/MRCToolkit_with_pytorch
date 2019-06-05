# coding:utf-8
import spacy
import re

class SpacyTokenizer(object):
    def __init__(self,fine_grained=False):
        self.nlp = spacy.load('en', disable=['parser','tagger','entity'])
        self.fine_grained = fine_grained

    def word_tokenizer(self, doc):
        if not self.fine_grained:
            doc = self.nlp(doc)
            tokens = [token.text for token in doc]
            token_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
            return tokens, token_spans
        sentence = doc
        tokens = []
        token_spans = []
        cur = 0
        pattern = u'-|–|—|:|’|\.|,|\[|\?|\(|\)|~|\$|/'
        for next in re.finditer(pattern, sentence):
            for token in self.nlp(sentence[cur:next.regs[0][0]]):
                if token.text.strip() != '':
                    tokens.append(token.text)
                    token_spans.append((cur + token.idx, cur + token.idx + len(token.text)))
            tokens.append(sentence[next.regs[0][0]:next.regs[0][1]])
            token_spans.append((next.regs[0][0], next.regs[0][1]))
            cur = next.regs[0][1]
        for token in self.nlp(sentence[cur:]):
            if token.text.strip() != '':
                tokens.append(token.text)
                token_spans.append((cur + token.idx, cur + token.idx + len(token.text)))
        return tokens, token_spans
