import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ✅ 모델 및 토크나이저 로드
base_model = "davidkim205/komt-mistral-7b-v1"
adapter_path = "output_mistral"
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

embedding_model = SentenceTransformer(embedding_model_name)

# ✅ 도서 데이터 로드
books_df = pd.read_json("book_dataset/cleanData.json")

# ✅ 유사도 기반 추천 함수
def extract_conditions(response):
    theme, book_type, age_list = "", "", []
    for line in response.split("\n"):
        if "theme:" in line:
            theme = line.split("theme:")[1].strip()
        elif "type:" in line:
            book_type = line.split("type:")[1].strip()
        elif "age:" in line:
            age_list = line.split("age:")[1].strip().split(",")
    return theme, book_type, age_list

def check_age_match(book_age, extracted_ages):
    if pd.isna(book_age):
        return False
    return any(age.strip() in book_age for age in extracted_ages)

def generate_response(prompt):
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
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def get_recommendations(user_input):
    # Step 1: 조건 추출
    cond_prompt = f"""### Instruction:\n다음 문장을 분석하여 도서 추천 조건을 추출하세요.\n\n### Input:\n{user_input}\n\n### Output:\n"""
    cond_output = generate_response(cond_prompt)
    theme, book_type, age_list = extract_conditions(cond_output)

    # Step 2: 필터링
    filtered = books_df[
        books_df['theme'].str.contains(theme, na=False) &
        books_df['type'].str.contains(book_type, na=False) &
        books_df['age'].apply(lambda x: check_age_match(x, age_list))
    ]
    if len(filtered) == 0:
        return [], "조건에 맞는 도서를 찾을 수 없습니다."

    book_texts = filtered['summary'] + ' ' + filtered['tags'] + ' ' + filtered['theme']
    book_vectors = embedding_model.encode(book_texts.tolist(), convert_to_numpy=True)
    query_vector = embedding_model.encode([user_input], convert_to_numpy=True)

    index = faiss.IndexFlatL2(book_vectors.shape[1])
    index.add(book_vectors)
    _, topk = index.search(query_vector, k=5)
    top_books = filtered.iloc[topk[0]].reset_index(drop=True)

    top_book = top_books.iloc[0]
    reason_prompt = f"""[질문] {user_input}\n[책 정보]\n제목: {top_book['title']}\n요약: {top_book['summary']}\n테마: {top_book['theme']}\n유형: {top_book['type']}\n대상 연령: {top_book['age']}\n[답변] 사용자의 요청에 따라 위 책을 추천하는 이유를 설명해주세요."""
    reason = generate_response(reason_prompt)

    return top_books.to_dict(orient="records"), reason

# ✅ 인터랙티브 터미널
if __name__ == "__main__":
    while True:
        user_input = input("\n📥 질문을 입력하세요 (종료하려면 'exit'): ")
        if user_input.strip().lower() == "exit":
            break

        books, reason = get_recommendations(user_input)
        if not books:
            print("😢 조건에 맞는 도서를 찾을 수 없습니다.")
            continue

        print("\n📚 추천 도서:", books[0]['title'])
        print("🧠 추천 이유:", reason)
        print("📌 기타 추천:")
        for book in books[1:]:
            print(" -", book['title'])
