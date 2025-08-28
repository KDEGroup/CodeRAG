from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model_id = "cache/pretrained_models/Qwen3-0.6B"
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

checkpoint = "cache/cceval/rerank/sft-good-no-think/checkpoint-2886"
peft_model = PeftModel.from_pretrained(base_model, checkpoint, device_map="auto", torch_dtype="auto")

save_model_path = "cache/cceval/rerank/sft-good-no-think-merged/Qwen3-0.6B-sft-eval-loss-0.0708"
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)

