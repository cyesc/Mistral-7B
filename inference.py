import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ✅ 모델 로드 설정
base_model = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_path = "/root/output"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("📦 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
print("✅ 튜닝된 모델 로드 완료!")

# ✅ 입력 루프
while True:
    user_input = input("\n📥 질문을 입력하세요 (종료하려면 'exit'): ")
    if user_input.lower().strip() == "exit":
        break

    # Prompt는 학습에 사용된 형식과 일치시켜야 함
    prompt = f"### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{user_input}\n\n### Output:\n"

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    tokens = outputs[0].cpu()
    decoded = tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # ✅ "### Output:" 뒷부분만 추출
    if "### Output:" in decoded:
        output_text = decoded.split("### Output:")[-1].strip()
    else:
        output_text = decoded.strip()

    print("\n🧠 모델 응답:\n", output_text)
