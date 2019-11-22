class Tokenizer(object):

    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        with open(self.vocab_file, 'r', encoding='utf-8') as stream:
            # sentences
            # self.vocab = {token.strip(): index for (index, token) in enumerate(stream)}

            # formulas
            self.vocab = {token: index for (index, token) in enumerate(stream)}
        self.vocab_size = len(self.vocab)

    def tokenize_seq(self, seq):
        output = seq.lower()
        output = output.split(" ")
        output = [sub_token for word in output for sub_token in self.tokenize_word(word)]
        return output

    def tokenize_word(self, word):
        word_len = len(word)
        characters = list(word)
        l = 0
        r = word_len
        sub_tokens = []
        # effort less ly
        while l < word_len:
            sub_str = None
            while r > l:
                sub_chr = characters[l:r]
                sub_str = "".join(sub_chr)
                if l > 0:
                    sub_str = "##" + sub_str
                if sub_str in self.vocab:
                    sub_tokens.append(sub_str)
                    l = r
                    r = word_len
                else:
                    r -= 1
            if sub_str is None:
                sub_tokens.append("[UNK]")
                break
        return sub_tokens

    def encode_word(self, word):
        return self.vocab[word]
