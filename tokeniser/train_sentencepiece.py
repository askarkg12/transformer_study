import sentencepiece as spm


VOCAB_SIZE = 20_000

options = dict(
    input="big_data/all_text.txt",
    input_format="text",
    model_prefix=f"tknz_{VOCAB_SIZE}",
    vocab_size=VOCAB_SIZE,
)

spm.SentencePieceTrainer.train(**options)
