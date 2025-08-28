from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from coderag.config import settings

model_id = settings.rerank.distill.training_base_model_path_or_name
base_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# data prepare
from coderag.rerank.supervise.common import SuperviseTrainingData, SuperviseTrainingDataItem
dataset_json = settings.rerank.distill.use_training_data_path
print(f"load training data from {dataset_json}")
with open(dataset_json) as f:
    data = SuperviseTrainingData.model_validate_json(f.read())
# data.data_list = data.data_list[:1000]
def generate_conversation(example: SuperviseTrainingDataItem):
    messages = example.messages.copy()
    messages.append({
        "role": "assistant",
        "content": example.expected
    })
    return messages
    
conversations = [generate_conversation(it) for it in data.data_list]
conversations = tokenizer.apply_chat_template(
    conversations,
    tokenize=False,
    enable_thinking=False
)
print(f"load {len(conversations)} conversations")
print(f"conversation[0]: {conversations[0]}")
combined_dataset = Dataset.from_dict({"text": conversations})
split = combined_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"training set num:{len(train_dataset)}")
print(f"eval set num:{len(eval_dataset)}")



peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
training_args = SFTConfig(
    dataset_text_field = "text",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps = 4, # Use GA to mimic batch size!
    warmup_steps = 5,
    num_train_epochs = 35, # Set this for 1 full training run.
    # max_steps = 1000,
    learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    report_to = "wandb", # Use this for WandB etc
    save_strategy="epoch",
    eval_strategy="epoch",
    output_dir=str(settings.rerank.distill.checkpoint_save_path),
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)
trainer.train()