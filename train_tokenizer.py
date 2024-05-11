from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import transformers
from tokenizers import SentencePieceBPETokenizer

paths = ['./plain_train.txt']
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
tk_tokenizer = SentencePieceBPETokenizer()
# Customize training
tk_tokenizer.train(files=paths, vocab_size=20000, min_frequency=2, special_tokens=special_tokens)
# convert


tokenizer = transformers.PreTrainedTokenizerFast(
    tokenizer_object=tk_tokenizer, 
    model_max_length=128, 
    special_tokens=special_tokens)
tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
# and save for later!
tokenizer.save_pretrained("./my_tokenizer")