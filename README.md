# blip-MedTrinity-peft

# 数据集
```python
from datasets import load_dataset

ds = load_dataset("UCSC-VLAA/MedTrinity-25M", "25M_demo")
DatasetDict({
    train: Dataset({
        features: ['image', 'id', 'caption'],
        num_rows: 161630
    })
})
```

# 模型加载
# Load model directly

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-base")
```

# 模型结构：
BlipForConditionalGeneration(
  (vision_model): BlipVisionModel(
    (embeddings): BlipVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    )
    (encoder): BlipEncoder(
      (layers): ModuleList(
        (0-11): 12 x BlipEncoderLayer(
          (self_attn): BlipAttention(
            (dropout): Dropout(p=0.0, inplace=False)
            (qkv): lora.Linear(
              (base_layer): Linear(in_features=768, out_features=2304, bias=True)
              (lora_dropout): ModuleDict(
                (default): Dropout(p=0.1, inplace=False)
              )
              (lora_A): ModuleDict(
                (default): Linear(in_features=768, out_features=16, bias=False)
              )
              (lora_B): ModuleDict(
                (default): Linear(in_features=16, out_features=2304, bias=False)
              )
              (lora_embedding_A): ParameterDict()
              (lora_embedding_B): ParameterDict()
              (lora_magnitude_vector): ModuleDict()
            )
            (projection): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): BlipMLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (text_decoder): BlipTextLMHeadModel(
    (bert): BlipTextModel(
      (embeddings): BlipTextEmbeddings(
        (word_embeddings): Embedding(30524, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (encoder): BlipTextEncoder(
        (layer): ModuleList(
          (0-11): 12 x BlipTextLayer(
            (attention): BlipTextAttention(
              (self): BlipTextSelfAttention(
                (query): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (key): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (value): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (output): BlipTextSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (crossattention): BlipTextAttention(
              (self): BlipTextSelfAttention(
                (query): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (key): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (value): lora.Linear(
                  (base_layer): Linear(in_features=768, out_features=768, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=768, out_features=16, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=16, out_features=768, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (lora_magnitude_vector): ModuleDict()
                )
                (dropout): Dropout(p=0.0, inplace=False)
              )
              (output): BlipTextSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
              )
            )
            (intermediate): BlipTextIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BlipTextOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    (cls): BlipTextOnlyMLMHead(
      (predictions): BlipTextLMPredictionHead(
        (transform): BlipTextPredictionHeadTransform(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (transform_act_fn): GELUActivation()
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
        (decoder): Linear(in_features=768, out_features=30524, bias=True)
      )
    )
  )
)
# 训练过程可视化
```python
import wandb

##使用注册时的API key登录wandb."Your API key

wandb.login(key = "21195b38ebd93abf1389611df531b21cc89bf294")

##建立wandb工程
run = wandb.init(project="fint-tune blip-image-captioning-base with open medical data",
                job_type="training",                
                anonymous="allow")
```
# 数据预处理
```python
def preprocess_function(examples, indices=None):  # 添加indices参数
    
    # 检查是否有已知的失败样本
    sample_ids = examples.get("id", [])
        
    
    # 处理图像（确保每张图像都是RGB模式）
    images = examples["image"]
    if not isinstance(images, list):
        images = [images]
    images = [img.convert("RGB") for img in images]
    
    # 处理文本（添加提示词）
    texts = examples["caption"]
    if not isinstance(texts, list):
        texts = [texts]
    texts = [f"Describing Medical Images：{caption}" for caption in texts]
    
    # 使用processor处理图像和文本
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="max_length",
        max_length=256,
        truncation=True
    )

    # print(inputs.keys())

    
    # 为训练准备标签
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs
    
   
split_ds = ds["train"].train_test_split(test_size=0.1)  # 10% 数据作为验证集

# 从训练集抽取3000个样本并预处理
train_subset = split_ds["train"].shuffle(seed=42).select(range(100000))

# 从验证集抽取1000个样本并预处理
validation_subset = split_ds["test"].shuffle(seed=42).select(range(1000))


print("预处理训练子集...")
processed_train_ds = train_subset.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=train_subset.column_names,
    num_proc=1,          # 单进程避免卡死
    load_from_cache_file=False,
    with_indices=False   # 关键：禁用索引传递
)

print("预处理验证子集...")
processed_val_ds = validation_subset.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=validation_subset.column_names,
    num_proc=1,          # 单进程避免卡死
    load_from_cache_file=False,
    with_indices=False   # 关键：禁用索引传递
)
LoRA微调参数配置
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


# 配置 LoRA 参数
lora_config = LoraConfig(
    target_modules=[
        "qkv",           # 视觉编码器的QKV矩阵
        "query",         # 文本解码器的query
        "key",           # 文本解码器的key
        "value",         # 文本解码器的value
    ],
    task_type=None,
    inference_mode=False,
    r=16,  # LoRA rank，适中以平衡性能和显存占用
    lora_alpha=32,  # 缩放因子
    lora_dropout=0.1  # Dropout 概率
)

# 将模型转换为 LoRA 模型
peft_model = get_peft_model(model, lora_config)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    hub_model_id="Logistic12/blip-MedTrinity-peft-1",
    report_to="tensorboard",
    run_name="medical-blip-lora"
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    fp16=True,
    remove_unused_columns=False,
    label_names=["labels"]
)
```
注意这里task_type要设置None，不然他会报错说数据里面有imputs_embeds
```python
peft_model.print_trainable_parameters()
```
trainable params: 2,359,296 || all params: 249,773,372 || trainable%: 0.9446

```python
from transformers import Trainer

# # # 定义 Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=processed_train_ds,
    eval_dataset=processed_val_ds,
    tokenizer=processor.tokenizer
)
trainer.train()

from transformers import pipeline

# 正确创建 pipeline 的方法（适用于最新版本 transformers）
pipe_peft = pipeline(
    "image-to-text",
    model=peft_model,
    tokenizer=processor.tokenizer,  # 明确指定 tokenizer
    image_processor=processor.image_processor,  # 使用 image_processor 替代 feature_extractor
    device="cuda" if torch.cuda.is_available() else "cpu"  # 指定设备
)

# 测试 pipeline
test_image = split_ds["test"][0]["image"].convert("RGB")  # 获取测试图片
result = pipe_peft(test_image)
print("Generated caption:", result[0]["generated_text"])

from huggingface_hub import notebook_login

notebook_login()
trainer.push_to_hub("Logistic12/blip-MedTrinity-peft-1")
```