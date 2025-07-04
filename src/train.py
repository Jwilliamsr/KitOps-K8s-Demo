import datasets, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

BASE = "google/flan-t5-small"
ds   = datasets.load_dataset("csv", data_files="data/tiny.csv")["train"]

tok = AutoTokenizer.from_pretrained(BASE)
def tok_fn(r):
    return tok(r["prompt"], text_target=r["answer"], truncation=True, max_length=128)
ds  = ds.map(tok_fn, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8))

args = Seq2SeqTrainingArguments(output_dir="ft-run", bf16=False, fp16=False,
                                per_device_train_batch_size=4, num_train_epochs=1)
data_collator = DataCollatorForSeq2Seq(tok, model=model)
trainer = Seq2SeqTrainer(model, args, train_dataset=ds, data_collator=data_collator)
trainer.train()

model.save_pretrained("fine-tuned")
tok.save_pretrained("fine-tuned")
print("✅ Fine-tune finished – weights in ./fine-tuned") 