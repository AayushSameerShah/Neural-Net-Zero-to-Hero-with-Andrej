import regex as re
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm

class RegexTokenizer:
    '''
    This is supposed to get a little crazy.
    
    Step 1: Split the text based on the regex pattern.
    Step 2: Now, we have the cleaned words.
    Step 3: Get their raw tokens individually.
    Step 4: Don't merge them yet, because it will nullify the step 1-3. 
    Step 4: Find pairs (stats) for each of the words - while keeping "common" stats across each.
    Step 5: Find the max repetative pair.
    Step 6: Replace that pair in each token group.
    '''
    def __init__(self):
        # initialize the defaut vocab
        self.vocab = {idx:bytes([idx]) for idx in range(256)}
        self.trained=False
        self.GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.GPT4_PATTERN_COMPILED = re.compile(self.GPT4_SPLIT_PATTERN)
    
    def find_most_repeated_pair(self, tokens, counter=None) -> Tuple[Tuple, int, Dict]:
        '''
        Now, this function is changed slightly as we will calculcate the 
        max when needed after this function call.
        
        Also, the `counter` can be passed and updated, and returned.
        Doing this will ensure, the global counter.
        '''
        counter = counter if counter is not None else defaultdict(int)
        for pair in zip(tokens, tokens[1:]):
            counter[pair] += 1
        return counter # will be useful when the counter=None passed.

    def replace_pair_with_new_token(self, tokens, pair, new_idx) -> List:
        new_tokens = [] # this will hold the copy for the new tokens
        idx = 0
        while idx < len(tokens):
            if idx < len(tokens) - 1 and (tokens[idx] == pair[0]) and (tokens[idx + 1] == pair[1]): # this is a match!
                new_tokens.append(new_idx)
                idx += 2
            else: # this is not a match
                new_tokens.append(tokens[idx])
                idx += 1
        return new_tokens
        
    def train(self, blob, vocab_size=None) -> None:
        '''
        This function will train the tokenizer based on the 
        training data given as text.
        
        1. blob: The data in text format that will be used as training
            of the tokenizer.
        
        2. vocab_size: This is "how many new tokens you want to generate"
            - `None` means indefinite; generate all combinations.
            - `int` means the number of merges.
        '''
        self.vocab_size = vocab_size
        
        # First split
        cleaned_text = self.GPT4_PATTERN_COMPILED.findall(blob)
        # Then create the tokens
        self.tokens = [list(map(int, word.encode("utf-8"))) for word in cleaned_text]
        
        
        new_idx = 255
        merges = {}
        for i in tqdm(range(vocab_size)):
            stats = defaultdict(int)
            for token_group in self.tokens:
                # pass the stats, which will be updated in place
                self.find_most_repeated_pair(token_group, stats)
            
            max_pair = max(stats, key=stats.get)
            max_count = stats[max_pair]
            
            if max_count > 1:
                new_idx += 1
                self.tokens = [self.replace_pair_with_new_token(token_group, max_pair, new_idx) for token_group in self.tokens]
                merges[max_pair] = new_idx
            else: # every pair is occuring for once only
                break
        self.total_merges = i+1
        
        ## The training is done now merge the stuff
        for pair, idx in merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.merges = merges   
        self.trained = True
        
    def encode(self, text):
        '''
        The goal of this function is to encode the given text into the 
        tokens that are acceptable by our `vocab`.
        
        So, we will need to keep encoding the tokens form the start (top)
        to the bottom.
        
        The `order` of the vocab **is not guerenteed** in the older versions
        of python, so we wil need to rely on the `idx`. The lower the idx
        is, the older that token is!
        '''
        
        if not self.trained:
            raise NotImplementedError("Please first train the tokenizer!")
        
        # tokens = text.encode("utf-8")
        split_words = self.GPT4_PATTERN_COMPILED.findall(text)
        split_tokens = [list(word.encode("utf-8")) for word in split_words]
        
        final_tokens = []
        for chunk in split_tokens:
            while len(chunk) >= 2:
                stats = self.find_most_repeated_pair(chunk)
                # now the goal is to get all pairs of the new tokens
                # we are not interested in the count, just the pairs
                # then check for each pair, if 
                pair_replace = min(self.merges, key=lambda x: stats.get(x, float("inf")))
                if pair_replace in stats:
                    chunk = self.replace_pair_with_new_token(chunk, 
                                                     pair_replace,
                                                     self.merges[pair_replace])
                else:
                    break
            final_tokens.extend(chunk)
        return final_tokens
    
    def decode(self, tokens):
        decoded_stream = [self.vocab[idx] for idx in tokens]
        text = b"".join(decoded_stream)
        return text.decode("utf-8")