from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

def create_tokenizer(texts):
    # Create a Byte-Level BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Apply normalization similar to GPT-2:
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    
    # Use ByteLevel pre-tokenization (this handles punctuation and spaces more robustly)
    tokenizer.pre_tokenizer = ByteLevel()
    
    # Set trainer with a vocabulary size close to GPT-2 (50257)
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size=50257)
    
    # Train the tokenizer on your text iterator
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer