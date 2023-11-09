from transformers import (
    DistilBertTokenizer,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForMaskedLM
)

tokenizer_base = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer_squad = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
tokenizer_sentiment = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

model_base = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
model_squad = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
model_sentiment = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# print(tokenizer_base.pad_token_id)
# print(tokenizer_squad.pad_token_id)
# print(tokenizer_sentiment.pad_token_id)

print(model_base.distilbert.embeddings.word_embeddings.weight[tokenizer_base.pad_token_id][:10])
print(model_base.distilbert.embeddings.word_embeddings.weight[tokenizer_base.sep_token_id][:10])
print(model_base.distilbert.embeddings.word_embeddings.weight[tokenizer_base.cls_token_id][:10])

print(model_squad.distilbert.embeddings.word_embeddings.weight[tokenizer_squad.pad_token_id][:10])
print(model_squad.distilbert.embeddings.word_embeddings.weight[tokenizer_squad.sep_token_id][:10])
print(model_squad.distilbert.embeddings.word_embeddings.weight[tokenizer_squad.cls_token_id][:10])

print(model_sentiment.distilbert.embeddings.word_embeddings.weight[tokenizer_sentiment.pad_token_id][:10])
print(model_sentiment.distilbert.embeddings.word_embeddings.weight[tokenizer_sentiment.sep_token_id][:10])
print(model_sentiment.distilbert.embeddings.word_embeddings.weight[tokenizer_sentiment.cls_token_id][:10])
