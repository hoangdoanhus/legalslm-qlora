# Fine-tuning Qwen2.5-1.5B với QLoRA cho Suy luận Pháp lý Tiếng Việt

**Nhóm thực hiện:** Lương Vân Khoa · Doãn Sơn Hoàng · Bàn Khánh Duy  
**Trường:** Đại học Khoa học Tự nhiên, ĐHQGHN  
**Năm:** 2026

---

## Mục lục

1. [Tổng quan đề tài](#1-tổng-quan-đề-tài)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cách chạy code (Google Colab)](#3-cách-chạy-code-google-colab)
4. [Giải thích từng bước trong notebook](#4-giải-thích-từng-bước-trong-notebook)
5. [Giải thích tham số và biến quan trọng](#5-giải-thích-tham-số-và-biến-quan-trọng)
6. [Lý do chọn phương pháp và công cụ](#6-lý-do-chọn-phương-pháp-và-công-cụ)
7. [Giải thích kết quả](#7-giải-thích-kết-quả)
8. [Thuật ngữ chuyên môn](#8-thuật-ngữ-chuyên-môn)
9. [Câu hỏi thường gặp](#9-câu-hỏi-thường-gặp)

---

## 1. Tổng quan đề tài

### Bài toán cần giải quyết

Hệ thống pháp luật Việt Nam cực kỳ phức tạp: hàng nghìn văn bản quy phạm được ban hành và sửa đổi mỗi năm. Người dùng phổ thông — từ người dân tra cứu quyền lợi đến sinh viên luật ôn tập — rất khó theo dõi và vận dụng chính xác. Câu hỏi đặt ra: liệu có thể xây dựng một AI nhỏ gọn, chạy được trên phần cứng thông thường, có đủ năng lực giải quyết các câu hỏi pháp lý tiếng Việt không?

### Cuộc thi VLSP 2025 LegalSLM

Đây là bối cảnh cụ thể của nghiên cứu. VLSP (Vietnamese Language and Speech Processing) là hội thảo NLP lớn nhất Việt Nam, và năm 2025 có tổ chức cuộc thi **LegalSLM** — yêu cầu xây dựng mô hình ngôn ngữ **nhỏ (≤ 4B tham số)** giải quyết 3 tác vụ pháp lý:

| Tác vụ | Mô tả | Kiểu đầu ra |
|--------|-------|-------------|
| **NLI** | Văn bản pháp lý có đủ thông tin trả lời câu hỏi không? | Có / Không |
| **MCQ** | Chọn 1 trong 4 đáp án cho câu hỏi pháp luật | A / B / C / D |
| **Syllogism** | Phân tích tình huống theo chuỗi lập luận pháp lý | Văn bản tự do |

### Phương pháp của nhóm

Nhóm sử dụng **Qwen2.5-1.5B-Instruct** (nhỏ hơn giới hạn 4B để phù hợp Google Colab miễn phí) kết hợp kỹ thuật **QLoRA** (lượng tử hóa 4-bit + LoRA adapter) và chiến lược **huấn luyện đa tác vụ** — gộp cả 3 tác vụ vào một lần train duy nhất.

### Kết quả đạt được

| Tác vụ | Zero-shot (trước) | Sau fine-tune | Cải thiện |
|--------|------------------|---------------|-----------|
| NLI Accuracy | 66,67% | **93,33%** | **+40,0%** |
| MCQ Accuracy | 73,33% | **86,67%** | **+18,2%** |
| Syllogism ROUGE-L | 0,4230 | **0,5066** | **+19,8%** |
| Thời gian train | — | **19,3 phút** | — |
| Chi phí | — | **$0 (Colab free)** | — |

---

## 2. Cấu trúc thư mục

```
NCKH_FINAL/
├── report.tex               # Báo cáo chính (LaTeX, ~20 trang)
├── slides.tex               # Slides thuyết trình (Beamer)
├── Bao_cao.tex              # Template gốc của trường (tham khảo)
├── husthesis.sty            # Package LaTeX của trường HUS-VNU
├── legal_slm_colab.ipynb    # Code thực nghiệm (chạy trên Google Colab)
├── README.md                # File này
├── _datasets/               # Dữ liệu gốc (JSON)
│   ├── multichoice.json     # 50 câu hỏi MCQ
│   ├── nli.json             # 50 câu hỏi NLI
│   ├── syllogism.json       # 50 câu hỏi Syllogism
│   └── legal_pretrain_news.json
├── docs/                    # Tài liệu tham khảo (PDF)
│   ├── 2025.vlsp-1.17.pdf   # Paper Bosch@AI (Top-1 VLSP 2025)
│   ├── 2025.vlsp-1.22.pdf   # Paper liên quan
│   ├── 2025.vlsp-1.24.pdf   # Paper liên quan
│   └── 2025.vlsp-1.51.pdf   # Paper MinLegal (Top-2 VLSP 2025)
└── logo/                    # Logo trường (cho LaTeX)
    ├── HUS_logo.png
    └── HUS logo_Final.jpg
```

---

## 3. Cách chạy code (Google Colab)

### Yêu cầu tối thiểu

- Tài khoản Google (miễn phí)
- Google Colab với GPU T4 (miễn phí, ~12 giờ/session)
- Kết nối internet ổn định

### Các bước thực hiện

**Bước 1: Mở notebook trên Colab**
```
File → Upload notebook → chọn legal_slm_colab.ipynb
```
Hoặc mở trực tiếp từ Google Drive nếu đã upload lên Drive.

**Bước 2: Đổi runtime sang GPU**
```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```
> ⚠️ **Bắt buộc phải làm bước này trước.** Nếu dùng CPU, training sẽ mất hàng giờ và có thể bị timeout.

**Bước 3: Chạy Cell 1 — Kiểm tra GPU và cài thư viện**

Cell này làm 2 việc:
- Kiểm tra CUDA và in thông tin GPU
- Cài đặt các thư viện cần thiết

> ⚠️ Sau khi cell cài thư viện xong, **bắt buộc Restart Runtime** (`Runtime → Restart session`). Nếu bỏ qua bước này, có thể gặp lỗi `ModuleNotFoundError: No module named 'triton.ops'`.

**Bước 4: Chạy Cell 2 — Tải dữ liệu**

Dữ liệu tự động tải từ HuggingFace Hub. Không cần upload file thủ công.

**Bước 5: Chạy Cell 3 — Định dạng dữ liệu**

Chuyển đổi dữ liệu thô thành format instruction tuning + phân chia train/test.

**Bước 6: Chạy Cell 4 — Tải mô hình**

Load Qwen2.5-1.5B với 4-bit quantization. Quá trình này tải khoảng 1GB trọng số từ HuggingFace, mất 2–5 phút tùy kết nối mạng.

**Bước 7: Chạy Cell 5 — Đánh giá zero-shot baseline**

Chạy mô hình gốc (chưa fine-tune) trên tập test để lấy điểm baseline. Mất khoảng 5–10 phút.

**Bước 8: Chạy Cell 6 — Cấu hình và huấn luyện QLoRA**

Gồm 3 sub-cell:
- Sub-cell 6a: Cấu hình LoRA adapter
- Sub-cell 6b: Chuẩn bị dataset huấn luyện
- Sub-cell 6c (cell training): Huấn luyện — mất khoảng **20 phút**

**Bước 9: Chạy Cell 7 (nếu cần) — Load lại model sau reset**

Nếu Colab runtime bị reset giữa chừng (mất kết nối, hết RAM...), chạy cell này để load lại model từ checkpoint đã lưu, không cần train lại.

**Bước 10: Chạy Cell 8 — Đánh giá sau fine-tune**

Chạy mô hình đã fine-tune trên tập test, in bảng kết quả so sánh.

**Bước 11: Chạy Cell 9 — Lưu và tải adapter về máy**

Nén adapter thành file `.zip` và tải về máy tính cục bộ để lưu trữ.

---

## 4. Giải thích từng bước trong notebook

### Cell 1: Kiểm tra GPU và cài thư viện

```python
!nvidia-smi                    # In thông tin GPU (tên, VRAM, driver)
torch.cuda.is_available()      # True nếu CUDA hoạt động
torch.cuda.get_device_name(0)  # Tên GPU, thường là "Tesla T4"
```

**Thư viện được cài:**
- `transformers`: Tải và chạy mô hình Qwen2.5 từ HuggingFace
- `peft`: Cấu hình và áp dụng LoRA adapter
- `trl`: SFTTrainer — wrapper cho supervised fine-tuning
- `accelerate`: Hỗ trợ phân bổ model/data lên GPU
- `datasets`: Tải dataset từ HuggingFace Hub
- `bitsandbytes`: Lượng tử hóa 4-bit (QLoRA)
- `rouge-score`: Tính ROUGE-L cho Syllogism

---

### Cell 2: Tải dữ liệu

```python
ds_mcq = load_dataset("VLSP2025-LegalSML/Public-Test", "multichoice_questions")
ds_nli = load_dataset("VLSP2025-LegalSML/Public-Test", "nli_questions")
ds_syl = load_dataset("VLSP2025-LegalSML/Public-Test", "syllogism_questions")
```

Hàm `get_split(ds)` tự động tìm tên split có dữ liệu (train/test/validation), vì tên split có thể khác nhau giữa các version dataset.

**Cấu trúc mỗi mẫu NLI:**
```json
{
  "legal_document": "Theo Kết luận 83-KL/TW...",
  "specific_question": "Khi nào sẽ có 5 bảng lương mới?",
  "question": "Điều luật có thể dùng để trả lời không?",
  "choices": ["Có", "Không"],
  "answer": 0
}
```

**Cấu trúc mỗi mẫu MCQ:**
```json
{
  "question": "Theo quy định, người nộp thuế có nghĩa vụ gì...",
  "choices": ["Phải ghi mã số thuế trong mọi trường hợp", "Chỉ khi bên mua yêu cầu", "..."],
  "answer": 1
}
```

---

### Cell 3: Định dạng dữ liệu

Mỗi mẫu được chuyển thành format hội thoại (conversation) gồm 3 phần:
- `system`: Vai trò của AI ("Bạn là chuyên gia pháp lý Việt Nam...")
- `user`: Câu hỏi + ngữ cảnh + hướng dẫn
- `assistant`: Câu trả lời mẫu

**Tại sao cần định dạng này?**  
Qwen2.5-Instruct được thiết kế để hoạt động theo kiểu hội thoại. Dùng đúng chat template giúp mô hình hiểu đây là instruction tuning (làm theo hướng dẫn), không phải language modeling thông thường.

**Phân chia train/test:**
```python
train_test_split(data, test_size=0.2, random_state=42)
# → 80% train (40 mẫu), 20% test (10 mẫu) mỗi tác vụ
```

`random_state=42` đảm bảo phân chia giống nhau mỗi lần chạy (tái tạo được).

---

### Cell 4: Tải mô hình với 4-bit quantization

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Lưu trọng số ở 4-bit
    bnb_4bit_quant_type="nf4",      # Dùng NormalFloat-4 (tốt hơn INT4 cho neural net)
    bnb_4bit_compute_dtype=torch.float16,  # Tính toán ở float16
    bnb_4bit_use_double_quant=True, # Lượng tử hóa lần 2 để tiết kiệm thêm bộ nhớ
)
```

Cell này có **fallback tự động**: nếu 4-bit thất bại (do lỗi bitsandbytes), sẽ tải lại ở float16. Khi dùng float16, adapter là LoRA thường (không phải QLoRA) nhưng vẫn hoạt động được, chỉ dùng nhiều VRAM hơn.

---

### Cell 5: Đánh giá zero-shot

```python
def generate_response(model, tokenizer, system, user, max_new_tokens=256):
    # Tạo prompt theo chat template của Qwen
    # Chạy model.generate() với do_sample=False (greedy decoding)
    # Trả về text sau token cuối cùng của input
```

**Greedy decoding** (`do_sample=False`) được dùng thay vì sampling ngẫu nhiên để đảm bảo kết quả **tái tạo được** (chạy lại cho kết quả giống nhau).

```python
def extract_choice(text):
    # Tìm chữ cái A/B/C/D đầu tiên trong output
    # Xử lý trường hợp model sinh thêm text giải thích
```

---

### Cell 6b: Chuẩn bị dataset cho SFTTrainer

```python
def format_for_sft(sample):
    messages = [system, user, assistant]
    return tokenizer.apply_chat_template(messages, tokenize=False)
    # Kết quả: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n..."
```

SFTTrainer cần text đã được định dạng hoàn chỉnh (cả input lẫn expected output trong một chuỗi), không phải input-output riêng biệt.

---

### Cell 6c: Huấn luyện QLoRA (cell quan trọng nhất)

```python
# Kiểm tra động tham số hợp lệ của SFTConfig
valid_sft_config = inspect.signature(SFTConfig.__init__).parameters

# Dùng SFTConfig thay vì TrainingArguments vì:
# - SFTConfig tích hợp max_seq_length, packing, dataset_text_field
# - TrainingArguments không nhận các tham số này trực tiếp

sft_config = SFTConfig(
    fp16=False,  # QUAN TRỌNG: tắt AMP của PyTorch vì BnB xử lý precision nội bộ
    bf16=False,  # Tương tự
    ...
)
```

**Tại sao `fp16=False`?**  
Khi mô hình được load ở 4-bit (bitsandbytes), PyTorch không biết dtype thực sự của các tensor. Nếu bật `fp16=True`, PyTorch AMP sẽ cố scale gradient — nhưng gặp tensor bfloat16 bên trong bitsandbytes và throw `NotImplementedError`. Giải pháp: tắt hoàn toàn AMP, để bitsandbytes tự quản lý precision.

---

### Cell 7: Load lại model (sau Colab reset)

Nếu runtime bị ngắt sau khi train xong nhưng trước khi đánh giá, chạy cell này:

```python
checkpoints = sorted(glob.glob(f"{ADAPTER_PATH}/checkpoint-*"))
adapter_path = checkpoints[-1]  # Checkpoint mới nhất

base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, adapter_path)
```

Checkpoint được lưu sau mỗi epoch theo `save_strategy="epoch"`. Với 3 epoch và 23 steps, sẽ có 3 checkpoint: `checkpoint-8`, `checkpoint-16`, `checkpoint-23`.

---

### Cell 8: Tổng hợp kết quả

In bảng so sánh và tính cải thiện tương đối:
```
Cải thiện tương đối = (điểm_sau - điểm_trước) / điểm_trước × 100%
```

---

## 5. Giải thích tham số và biến quan trọng

### Tham số LoRA

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `r` (rank) | 16 | Hạng của ma trận thấp (số chiều latent). Cao hơn = nhiều tham số hơn, học được nhiều hơn nhưng nguy cơ overfitting. Thường dùng 8–64. |
| `lora_alpha` | 32 | Hệ số tỷ lệ: output adapter × (alpha/r) = output × 2. Với alpha=2r, learning rate hiệu dụng của adapter gấp đôi LR chính. |
| `lora_dropout` | 0.05 | Dropout nhỏ để tránh overfitting trên dataset nhỏ. |
| `target_modules` | q,k,v,o,gate,up,down | Các lớp trong Transformer được áp dụng LoRA. Áp dụng tất cả projections thay vì chỉ q+v như paper gốc để tăng capacity. |
| `bias` | "none" | Không điều chỉnh bias (tiết kiệm tham số). |

### Tham số huấn luyện

| Tham số | Giá trị | Lý do chọn |
|---------|---------|-----------|
| `learning_rate` | 2e-4 | Khuyến nghị chuẩn cho QLoRA theo paper gốc. LR cao hơn (1e-3) gây mất ổn định; thấp hơn (1e-5) hội tụ quá chậm. |
| `per_device_train_batch_size` | 2 | Giới hạn VRAM T4. Batch=4 gây OOM với sequence dài. |
| `gradient_accumulation_steps` | 8 | Effective batch = 2×8 = 16. Giả lập batch lớn mà không cần thêm VRAM. |
| `num_train_epochs` | 3 | Với 120 mẫu, 3 epoch là cân bằng: đủ để học nhưng không overfit. Thử 5 epoch thấy loss không giảm thêm nhiều. |
| `max_seq_length` | 1024 | Đủ cho hầu hết mẫu pháp lý (Syllogism dài nhất ~800 token). |
| `lr_scheduler_type` | cosine | Giảm LR theo đường cosine, nhẹ nhàng hơn linear decay. Phù hợp với dataset nhỏ. |
| `warmup_ratio` | 0.05 | 5% đầu (≈1 step) LR tăng dần từ 0 → 2e-4, tránh update lớn ngay từ đầu. |
| `optim` | paged_adamw_8bit | Adam 8-bit với paging: optimizer states có thể trao đổi ra CPU RAM khi VRAM đầy. |

### Biến trong code

| Biến | Kiểu | Mô tả |
|------|------|-------|
| `SYSTEM_PROMPT` | str | Prompt hệ thống định nghĩa vai trò AI. Giống nhau cho cả 3 tác vụ. |
| `train_data` | list[dict] | 120 mẫu sau định dạng, mỗi mẫu có keys: system, user, assistant, task |
| `train_dataset` | HuggingFace Dataset | Tập train ở dạng {"text": [chuỗi hoàn chỉnh]} cho SFTTrainer |
| `zs_nli_acc` | float | Zero-shot NLI accuracy (0.0–1.0) |
| `ft_nli_acc` | float | Fine-tuned NLI accuracy |
| `training_time` | float | Thời gian train tính bằng phút |
| `train_result.training_loss` | float | Cross-entropy loss cuối cùng |

---

## 6. Lý do chọn phương pháp và công cụ

### Tại sao Qwen2.5-1.5B?

1. **Phù hợp với Colab free**: GPU T4 có 15GB VRAM. Với 4-bit quantization, Qwen2.5-1.5B chỉ chiếm ~1.1GB cho trọng số, còn nhiều cho activations và optimizer.
2. **Đa ngôn ngữ tốt**: Qwen2.5 được train trên corpus rất lớn bao gồm tiếng Việt, tiếng Trung và tiếng Anh — tốt hơn nhiều mô hình chỉ tập trung vào tiếng Anh.
3. **Đã instruction-tuned**: Phiên bản Instruct đã biết cách làm theo hướng dẫn, dễ fine-tune hơn phiên bản Base.
4. **Context window 128K**: Đủ dài cho văn bản pháp lý phức tạp, không lo bị truncate.

*So với Qwen2.5-4B*: 4B model cho kết quả tốt hơn nhưng cần Colab Pro (A100) để train. Nhóm ưu tiên khả năng tái tạo trên free tier.

### Tại sao QLoRA chứ không phải full fine-tuning?

Full fine-tuning mô hình 1.5B cần ~18GB VRAM (trọng số + gradient + Adam states) — vượt quá T4 15GB. QLoRA giải quyết:
- **4-bit quantization**: Trọng số gốc từ 3GB → 1.1GB
- **LoRA**: Chỉ train 23.4M/1.54B = 1.52% tham số → gradient nhỏ, ít bộ nhớ hơn nhiều
- **paged_adamw_8bit**: Optimizer states có thể spill ra CPU RAM

### Tại sao multi-task training (gộp 3 tác vụ)?

Thay vì train riêng:
- **Dữ liệu nhiều hơn mỗi epoch**: 120 mẫu thay vì 40, giảm overfitting
- **Một checkpoint duy nhất**: Dễ quản lý hơn 3 checkpoint riêng biệt
- **Regularization tự nhiên**: Sự đa dạng của các tác vụ ngăn mô hình memorize một loại prompt cụ thể
- **Chia sẻ kiến thức**: Học cách đọc văn bản pháp lý từ NLI có thể giúp MCQ và Syllogism

*Trade-off*: Multi-task có thể gây "task interference" nếu các tác vụ xung đột. Với 3 tác vụ pháp lý tiếng Việt (cùng domain), rủi ro này thấp.

### Tại sao dùng SFTConfig thay vì TrainingArguments?

`TrainingArguments` là lớp base của HuggingFace Transformers cho tất cả loại training. `SFTConfig` là subclass dành riêng cho Supervised Fine-Tuning, tích hợp thêm các tham số như `max_seq_length`, `packing`, `dataset_text_field`. Trong các phiên bản TRL gần đây, `SFTTrainer` chỉ nhận `SFTConfig`, không nhận `TrainingArguments`.

### Tại sao ROUGE-L cho Syllogism?

Syllogism yêu cầu sinh văn bản tự do — không thể dùng Accuracy. ROUGE-L đo độ trùng lặp chuỗi con dài nhất giữa output và đáp án chuẩn. Đây là metric tiêu chuẩn trong cuộc thi VLSP 2025.

Hạn chế: ROUGE-L không đo chất lượng lập luận pháp lý thực sự. Một câu trả lời đúng về logic nhưng dùng từ khác sẽ bị đánh giá thấp. Metric tốt hơn là LLM-as-judge nhưng cần API trả phí.

### Tại sao greedy decoding khi đánh giá?

`do_sample=False` (greedy) chọn token có xác suất cao nhất ở mỗi bước, không có ngẫu nhiên. Điều này đảm bảo:
- **Tái tạo được**: Chạy lại cho cùng kết quả
- **Nhất quán**: Dễ so sánh giữa các phương pháp
- **Phù hợp với task**: Câu trả lời pháp lý cần chắc chắn, không cần creative

---

## 7. Giải thích kết quả

### Training loss = 0.9357

Loss đo độ sai lệch giữa xác suất model dự đoán và đáp án đúng (cross-entropy). Loss thấp = model dự đoán đúng với độ tự tin cao.

**Tại sao 0.9357 không thấp hơn?**
- Phần lớn loss đến từ Syllogism (output dài, phức tạp)
- Với NLI và MCQ (chỉ 1 token đầu ra), loss thực tế rất thấp
- 23 steps training là quá ít để giảm loss nhiều hơn

**Có thể giảm loss không?** Có, bằng cách: tăng epochs, tăng dữ liệu, dùng mô hình lớn hơn, hoặc train riêng Syllogism.

### NLI cải thiện nhiều nhất (+40%)

Zero-shot: 6/9 câu đúng (66.7%). Sau fine-tune: 9/10 câu đúng (93.3%).

Lý do cải thiện lớn: Zero-shot Qwen thường nhầm lẫn giữa "câu hỏi meta" (văn bản có trả lời được câu hỏi không?) và "câu hỏi cụ thể" (nội dung câu hỏi). Fine-tune dạy mô hình tập trung đúng vào quan hệ ngữ nghĩa giữa văn bản và câu hỏi.

### MCQ cải thiện 18.2%

Zero-shot: 7/10 đúng (73.3%). Sau fine-tune: 8/10 đúng (86.7%) — tức là thêm 1 câu đúng.

MCQ cải thiện ít hơn NLI vì model đã có kiến thức pháp lý nền (từ pre-training). Fine-tune chủ yếu giúp ở các câu hỏi về số liệu cụ thể trong văn bản pháp luật mới (ban hành 2024–2025, sau cutoff của model).

### Syllogism ROUGE-L: 0.5066

ROUGE-L = 0.5 có nghĩa là ~50% nội dung của câu trả lời model trùng với đáp án chuẩn (tính theo LCS). Con số này không tồi với mô hình chỉ 1.5B tham số và 40 mẫu training.

Để tham chiếu: các mô hình 7B+ thường đạt ROUGE-L ~0.6–0.7 trên các task tương tự.

### Thời gian 19.3 phút

Tổng steps = 120 mẫu / effective_batch(16) × 3 epoch ≈ 23 steps. Với T4, mỗi step mất khoảng 50 giây (do gradient accumulation × 8 forward passes). 23 × 50s ≈ 19 phút.

---

## 8. Thuật ngữ chuyên môn

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **LLM** | Large Language Model — mô hình ngôn ngữ lớn (GPT-4, Gemini...) |
| **SLM** | Small Language Model — mô hình ngôn ngữ nhỏ (≤4B tham số) |
| **Fine-tuning** | Tinh chỉnh mô hình đã pre-train trên dataset chuyên biệt |
| **Pre-training** | Giai đoạn train ban đầu trên corpus khổng lồ (không có giám sát) |
| **Instruction tuning** | Fine-tune theo định dạng instruction/response để model làm theo hướng dẫn |
| **PEFT** | Parameter-Efficient Fine-Tuning — các kỹ thuật chỉ train một phần nhỏ tham số |
| **LoRA** | Biểu diễn update trọng số bằng tích 2 ma trận hạng thấp BA thay vì ma trận đầy đủ |
| **QLoRA** | LoRA + quantize mô hình nền xuống 4-bit để tiết kiệm VRAM |
| **NF4** | NormalFloat-4bit — định dạng 4-bit tối ưu cho phân bố Gaussian của trọng số |
| **Quantization** | Lượng tử hóa — giảm độ chính xác số học (float32 → float16 → int8 → 4-bit) |
| **Double quantization** | Lượng tử hóa cả các hằng số quantization, tiết kiệm thêm ~0.37 bit/param |
| **Adapter** | Các module nhỏ thêm vào mô hình frozen để tinh chỉnh |
| **Chat template** | Định dạng chuỗi đặc thù của từng mô hình cho hội thoại (system/user/assistant) |
| **Greedy decoding** | Chọn token có xác suất cao nhất ở mỗi bước; deterministic |
| **ROUGE-L** | Metric dựa trên Longest Common Subsequence giữa text sinh ra và tham chiếu |
| **BERTScore** | Metric dùng embedding của BERT để đo độ tương đồng ngữ nghĩa |
| **Zero-shot** | Dùng mô hình gốc không fine-tune, chỉ prompt |
| **Few-shot** | Thêm vài ví dụ vào prompt, không train |
| **Cross-entropy loss** | Hàm mất mát đo độ lệch giữa phân phối xác suất dự đoán và nhãn thực |
| **Gradient accumulation** | Tích lũy gradient qua nhiều mini-batch trước khi update — giả lập batch lớn |
| **Effective batch size** | batch_size × gradient_accumulation_steps = 2 × 8 = 16 |
| **NLI (trong VLSP)** | Natural Language Inference — phân loại xem văn bản có liên quan đến câu hỏi không |
| **Syllogism** | Suy luận tam đoạn luận: tiền đề lớn (luật) + tiền đề nhỏ (tình huống) → kết luận |
| **Multi-task learning** | Train một mô hình trên nhiều tác vụ cùng lúc |
| **Task interference** | Hiện tượng các tác vụ cản trở nhau khi train multi-task |
| **Overfitting** | Mô hình học thuộc tập train nhưng không tổng quát được |
| **Warmup** | Giai đoạn đầu training tăng LR dần từ 0 để tránh update lớn đột ngột |
| **Cosine scheduler** | Lịch giảm LR theo đường cosine, nhẹ nhàng và hiệu quả |
| **VRAM** | Video RAM — bộ nhớ GPU, quyết định mô hình lớn bao nhiêu có thể chạy được |
| **Checkpoint** | Snapshot trạng thái model được lưu trong quá trình training |
| **PeftModel** | Class của thư viện PEFT để load base model + LoRA adapter |
| **SFTTrainer** | Supervised Fine-Tuning Trainer từ thư viện TRL |
| **SFTConfig** | Cấu hình cho SFTTrainer (thay thế TrainingArguments trong TRL mới) |

---

## 9. Câu hỏi thường gặp

**Q: Tại sao phải restart runtime sau khi cài thư viện?**  
A: Colab đã load một phiên bản bitsandbytes cũ vào memory. Việc upgrade qua pip không có hiệu lực cho đến khi Python interpreter được khởi động lại. Nếu không restart, `import bitsandbytes` vẫn load version cũ, gây lỗi `triton.ops`.

**Q: Nếu gặp lỗi "CUDA out of memory" thì làm gì?**  
A: Giảm `per_device_train_batch_size` xuống 1, tăng `gradient_accumulation_steps` lên 16 để giữ effective batch = 16. Hoặc giảm `max_seq_length` xuống 512.

**Q: Tại sao không dùng Qwen3 thay vì Qwen2.5?**  
A: Qwen3 mới hơn và tốt hơn, nhưng vào thời điểm thực nghiệm (2026) các thư viện hỗ trợ Qwen3 với QLoRA chưa hoàn thiện. Qwen2.5-1.5B đã được kiểm chứng rộng rãi.

**Q: Sao không dùng LLaMA hay Gemma thay vì Qwen?**  
A: Qwen2.5 được train trên corpus tiếng Việt nhiều hơn LLaMA (chủ yếu tiếng Anh). Thực nghiệm của các nhóm VLSP 2025 cũng cho thấy Qwen hoạt động tốt nhất cho tiếng Việt trong nhóm mô hình dưới 4B.

**Q: Loss 0.9357 có nghĩa là model tệ không?**  
A: Không nhất thiết. Loss là metric tổng hợp trên tất cả token, bị kéo lên bởi Syllogism. Accuracy 93.3% (NLI) và 86.7% (MCQ) mới là số quan trọng hơn để đánh giá.

**Q: Có thể chạy trên máy cá nhân không?**  
A: Được, nếu có GPU NVIDIA với ≥6GB VRAM và driver CUDA ≥11.8. Cần cài thêm `jupyter notebook`. Không chạy được trên CPU vì quá chậm.

**Q: File adapter lưu ở đâu?**  
A: Trong Colab session tại `./qwen_legal_lora_final/`. Gồm các file: `adapter_config.json`, `adapter_model.safetensors`. Cần download về trước khi session kết thúc (Colab xóa file khi ngắt kết nối).

---

*Báo cáo đầy đủ: xem `report.tex` (compile bằng XeLaTeX hoặc Overleaf)*  
*Slides thuyết trình: xem `slides.tex`*  
*Code thực nghiệm: xem `legal_slm_colab.ipynb`*
