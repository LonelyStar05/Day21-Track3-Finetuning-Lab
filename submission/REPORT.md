# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Tú Tài  **MSSV**: 2A202602455  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14,6 GB khả dụng, sm_75 → fp16)

> Mọi con số dưới đây lấy từ `results/` của lần chạy đầy đủ (`eval_limit = null`,
> `smoke_mode = false`, n_target = 50, n_regression = 15).

---

## 0. Lựa chọn và lý do

| | Chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định của tier T4) | Model lớn nhất vừa T4 ở bf16/fp16 LoRA (~10 GB); không cần QLoRA nên phép so sánh `qlora` ở NB4 là một phép đo, không phải một sự bắt buộc. |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON 4 trường | Cả 4 nhóm điểm đều chấm khách quan (không cần LLM judge); checksum tập eval đã đóng băng nên kết quả kiểm chứng được. Chạy corpus gốc trước để hiểu pipeline, đúng như README gợi ý. |
| `OPTIMIZED_PROMPT` | **Không sửa** — SHA `719e74d3b6232053` | (b) đã hơn (a) rất xa (0.765 vs 0.000); không có lý do làm mạnh thêm, và tuyệt đối không làm yếu đi. |

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (corpus mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (giá trị của tier) — p95 đo được là **98** token, gợi ý **256** *(results/token_stats.json)* |
| Token stats | n=250 · mean 93.1 · p50 93 · p95 98 · p99 100 · max 101 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / **30** (cả 4 run) |
| Precision | fp16 + GradScaler (T4 không có bf16) |

**Về `max_length`:** tôi giữ 1024 của tier thay vì 256 gợi ý. Lý do: mẫu dài nhất chỉ
101 token nên **không mẫu nào bị cắt** ở cả hai giá trị; và với `batch=1` không có
padding, nên `max_length` chỉ là trần — chi phí tính toán phụ thuộc độ dài thật (~93
token), không phụ thuộc trần. Đổi sang 256 không làm thay đổi kết quả. Nếu tăng batch
size thì 256 mới đáng đặt, vì khi đó padding bắt đầu tốn tiền.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`:
`ok = true`, `body_present = true`, verdict *"reasoning preserved — safe to train on
traces"*. Chuỗi render vẫn chứa nguyên `<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>`.
Corpus này không có trace (câu trả lời là JSON trần), nên điều này không ảnh hưởng run
chính; nhưng nó nghĩa là thí nghiệm B3 (`masked-think` / `response-only`) **làm được**
trên base này nếu có dữ liệu chứa trace.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `mask_mode` | `assistant-only` |
| `supervised_fraction` | **0.4149** (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn **được tính loss** (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị mask** (`masked_preview`):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

```

Đọc ngược hai đoạn: system prompt, ticket và cả header `assistant` + `<think>` mở đều
nằm ngoài loss; loss chỉ tính trên JSON và `<|im_end|>` (tín hiệu dừng — đó là lý do
`format = 1.0` ở bản fine-tune). `supervised_fraction = 0.41`, rất xa ngưỡng 0.95 của
lỗi "tính loss cả prompt".

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) | n |
|---|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3705.1 | 50 |
| (b) base + optimized prompt | **0.765** | 0.7911 | 1.000 | 1153.6 | 50 |
| (c) LoRA fine-tune *(NB5, naive prompt)* | **0.970** | **0.5222** | 1.000 | 1604.0 | 50 |

**(b) có thật sự mạnh hơn (a) không?** **Có**, cách biệt rất lớn: target 0.765 vs
0.000, format 1.000 vs 0.000. Với prompt ngắn "Phân loại ticket sau." base model không
trả về JSON đủ 4 khoá ở bất kỳ mẫu nào, và latency cao gấp ~3 lần (3705 ms vs 1154 ms)
— dấu hiệu nó sinh dài hơn nhiều thay vì một object JSON ngắn. Regression của (a) và (b) bằng
nhau (0.7911) là đúng thiết kế: nhóm regression gửi `system=None`, nên không phụ thuộc
prompt triage.

Tôi **không sửa** `OPTIMIZED_PROMPT` (SHA `719e74d3b6232053` khớp bản gốc).

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | latency ms | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32 464 896 | 1e-4 | 0.6258 | **0.970** | 1.000 | 1604.0 | 447.2 | 8.78 |
| `attn_only` | q,v (attn-only) | 283 | 32 456 704 | 1e-4 | **0.5373** | **0.970** | 1.000 | 1020.6 | 296.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32 464 896 | **1e-5** | 1.5702 | **0.000** | 0.000 | 5989.2 | 441.9 | 8.78 |
| `qlora` | text-linear, base 4-bit | 16 | 32 464 896 | 1e-4 | 0.7058 | **0.940** | 1.000 | 2035.0 | 546.3 | **3.86** |

Kiểm tra tính công bằng:

- **Ngân sách tham số:** `attn_only` lệch `correct` |32 456 704 − 32 464 896| / 32 464 896
  = **0.025 %** (< 5 %), nhờ `matched_rank()` nâng r từ 16 lên 283.
- **Cùng số step:** cả bốn run `max_steps = 30`.
- **Một biến mỗi run:** `attn_only` chỉ đổi *vị trí* (và rank để giữ ngân sách);
  `wrong_lr` chỉ đổi *LR* (1e-4 → 1e-5); `qlora` chỉ đổi *độ chính xác của base*
  (16-bit → 4-bit). Mọi thứ khác — dữ liệu, mask, step, alpha/r = 2 — giữ nguyên.

**Hai cách xếp hạng cho hai thứ tự khác nhau:**

- Theo train loss (thấp = "tốt"): `attn_only` (0.537) > `correct` (0.626) > `qlora` (0.706) > `wrong_lr` (1.570)
- Theo target (NB5 §4): `correct` = `attn_only` (0.970) > `qlora` (0.940) > `wrong_lr` (0.000)

Nhìn train loss, ta sẽ kết luận `attn_only` là cấu hình tốt nhất. Trên tác vụ thật, nó
chỉ **hoà**. Đây chính là Lỗi #3: chấm bằng chỉ số thay thế.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**

Trên tập target `attn_only` **hoà** `correct` đúng tuyệt đối: cả hai 0.970, format
1.000. Nhưng train loss của nó thấp hơn rõ rệt (0.537 vs 0.626), nên thứ tự theo loss
và theo target **không giống nhau**: loss nói `attn_only` thắng, tác vụ nói hoà. Loss
thấp hơn ở đây nhiều khả năng là adapter r=283 dồn vào ít module đã khớp tập train
225 mẫu chặt hơn — một dạng ghi nhớ — chứ không phải học tác vụ tốt hơn. Về rank so với
vị trí: với cùng 32,5 triệu tham số, tăng rank lên ~18 lần mà chỉ gắn vào q,v không mua
thêm được điểm nào; ngược lại, rải r=16 lên mọi lớp tuyến tính cũng không thua. Trên một
tác vụ hẹp và dễ như triage JSON 4 trường, **cả hai đều đã chạm trần (~0.97)**, nên thí
nghiệm này không phân định được "vị trí thắng rank" như deck §11.2 khẳng định — nó chỉ
chứng minh được rằng *rank không phải đòn bẩy*. Điểm đáng chú ý là `attn_only` lại
nhanh hơn khi suy luận (1021 ms vs 1604 ms) và train nhanh hơn (296 s vs 447 s) vì
adapter chạm ít module hơn — một lợi thế thật nếu không merge adapter. Để phân định
được vị trí với rank, cần một tác vụ khó hơn (điểm chưa bão hoà) hoặc ít step hơn.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

`wrong_lr` dùng LR 1e-5 (thang full fine-tune) thay vì 1e-4. Loss cuối trung bình của
nó là **1.570**, gấp ~2,5 lần `correct` (0.626). Nhưng đường loss *không* phẳng: theo
log NB4 nó giảm đều từ 2.163 xuống 1.119 trong 30 step, và `mean_token_accuracy` cuối đạt
~0.79. Chỉ nhìn đường cong đó, kết luận tự nhiên sẽ là "model đang học, chỉ chậm hơn —
train thêm vài epoch là đuổi kịp", hoặc tệ hơn "0.79 token accuracy là khá ổn". Thực tế
trên tác vụ là **target 0.000, format 0.000**: không một mẫu nào ra JSON parse được — y
hệt baseline (a) của base model với cùng prompt ngắn (0.000 / 0.000), và latency
5989 ms cho thấy nó sinh văn xuôi dài như base. Tức là với LR quá nhỏ, adapter chưa đủ
lớn để lật hành vi mặc định của model ở token đầu tiên; token accuracy cao đến từ
teacher forcing (được mớm sẵn phần JSON phía trước), còn khi tự sinh thì model đi lạc
ngay từ đầu. Đây là kết quả rõ nhất trong lab: **LR là đòn bẩy mạnh nhất** — chỉ sai một
bậc độ lớn là từ 0.970 xuống 0.000, trong khi đổi vị trí hay lượng tử hoá chỉ dao động
0.00–0.03.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**

`qlora` dùng **3.86 GB** peak VRAM so với **8.78 GB** của `correct` — tiết kiệm 4.92 GB,
tức **56 %**. Cái giá: target giảm 0.030 (0.940 vs 0.970, tương đương 6 trường sai thêm
trên 200 trường), train loss cao hơn (0.706 vs 0.626), train chậm hơn 22 % (546 s vs
447 s, do phải giải lượng tử hoá trọng số ở mỗi bước) và suy luận chậm hơn 27 % (2035 ms
vs 1604 ms). Số đo **ủng hộ có điều kiện** khuyến nghị của nhà cung cấp: trên T4, bản
16-bit đã vừa (8.78 / 14.6 GB), nên trả cả độ chính xác lẫn tốc độ để tiết kiệm VRAM mà
mình không cần là lỗ. Tuy vậy mức suy giảm 0.03 là nhỏ (và với n=50 thì khoảng tin cậy
không hẹp), nên nếu phần cứng không chứa nổi bản 16-bit — ví dụ muốn chạy bản 9B trên
T4 — QLoRA vẫn là lựa chọn hợp lý chứ không phải "không dùng được".

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**
`target Δ = +0.205` · `regression Δ = −0.269` · `valid_trace_rate = 0.00`

> - general capability regressed by 0.269 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.

**Diễn giải.** Bản fine-tune thắng tác vụ chính một cách thuyết phục: target 0.970 so
với 0.765 của prompt tối ưu (+0.205), format giữ 1.000, và nó đạt điều đó *chỉ với prompt
ngắn 5 chữ* — hành vi đã chuyển từ prompt vào trọng số, đúng mục đích của fine-tune. Nhưng
nó trượt cổng ở nhóm thứ hai: điểm regression rơi từ 0.791 xuống 0.522, mất 0.269 trong
khi ngưỡng cho phép chỉ 0.020 — gấp hơn 13 lần. Đó là **quên thảm hoạ** (deck §6.3).
Nguyên nhân nằm ở dữ liệu chứ không ở cấu hình LoRA: cả 225 mẫu train đều cùng một dạng
"ticket → JSON", không có một mẫu nào dạy model rằng câu hỏi khác thì phải trả lời khác.
Sau 30 step, model học được một quy tắc quá rộng — "đầu vào nào cũng là ticket" — và bắt
đầu trả lời câu hỏi kiến thức phổ thông theo khuôn triage. Ghi chép mô phỏng của chính
repo (`SIMULATION-FINDINGS.md`) mô tả đúng hiện tượng này ở bản 0.8B (model trả lời "thủ
đô Việt Nam?" bằng JSON triage), và đã loại trừ giả thuyết "chỉ do khác hình dạng prompt".
Trên bản 4B mức sụp nhẹ hơn nhiều (0.522 so với 0.067 ở 0.8B) — model lớn giữ được nhiều
năng lực chung hơn — nhưng vẫn quá xa ngưỡng.

`valid_trace_rate = 0.00` không phải lỗi: dữ liệu train không có trace suy luận và
template đóng sẵn `<think></think>` rỗng, nên không có gì để đo ở đây.

Kết luận của phán quyết: với dữ liệu hiện tại, **không nên deploy** bản fine-tune này
như một model đa dụng. Cách sửa đúng là trộn 1–5 % dữ liệu phổ thông (replay) vào tập
train rồi đo lại — **không** phải nới ngưỡng hay sửa tập eval.

---

## 6. Định tính — có cả ca THUA

`qualitative.json` xếp 50 mẫu theo điểm của bản fine-tune. Điểm thấp nhất là 0.75, và vì
tổng target 0.970 = 194/200 trường đúng, có đúng 6 ticket mỗi ticket sai một trường.
Dưới đây là 3 ca tệ nhất và 3 ca tốt nhất mà NB5 in ra. (Lab không lưu dự đoán từng mẫu
của baseline (b), nên cột "FT thua/thắng" so với **nhãn đúng**.)

| # | i | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Điểm | Nhận xét |
|---|---|---|---|---|---|---|
| 1 | 3 | "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | hoan_tien · **trung_binh** · bình giữ nhiệt · tich_cuc | 0.75 | ❌ **FT thua** — sai urgency |
| 2 | 5 | "…nồi chiên không dầu… Thiếu phụ kiện. **Khi nào tiện.** Cho tôi hỏi." | san_pham_loi · **thap** · nồi chiên không dầu · trung_tinh | san_pham_loi · **trung_binh** · nồi chiên không dầu · trung_tinh | 0.75 | ❌ **FT thua** — sai urgency |
| 3 | 12 | "…áo khoác gió… Bị lỗi. **Khi nào tiện.** Cảm ơn shop nhiều." | san_pham_loi · **thap** · áo khoác gió · tich_cuc | san_pham_loi · **trung_binh** · áo khoác gió · tich_cuc | 0.75 | ❌ **FT thua** — sai urgency |
| 4 | 47 | "…ốp lưng điện thoại… Shipper không gọi. Hỏi cho biết thôi. Shop hỗ trợ tốt." | van_chuyen · thap · ốp lưng điện thoại · tich_cuc | khớp cả 4 trường | 1.00 | ✅ FT thắng |
| 5 | 48 | "…ốp lưng điện thoại… Giá bao nhiêu. Mong shop phản hồi." | hoi_thong_tin · trung_binh · ốp lưng điện thoại · trung_tinh | khớp cả 4 trường | 1.00 | ✅ FT thắng |
| 6 | 49 | "…ốp lưng điện thoại… Sai màu. Sớm nhé. Shop xem giúp." | san_pham_loi · trung_binh · ốp lưng điện thoại · trung_tinh | khớp cả 4 trường | 1.00 | ✅ FT thắng |

*(Dự đoán của ca 1–3: NB5 in rút gọn 90 ký tự, thấy urgency = `trung_binh`; điểm 0.75 nghĩa
là đúng 1 trường sai, nên 3 trường còn lại khớp nhãn.)*

Ngoài ra, ca thua lớn nhất **không nằm trong bảng trên** mà ở nhóm regression: trên 15
câu hỏi phổ thông, bản fine-tune giảm từ 0.791 xuống 0.522 keyword recall (xem §5).

**Mẫu chung ở các ca FT thua.** Cả ba lỗi đều cùng một trường (`urgency`), cùng một
hướng (`thap` → `trung_binh`), và cả ba ticket đều chứa cụm **"Khi nào tiện"**. Tôi kiểm
tra corpus: trong tập train, cả **35/35** ticket có "Khi nào tiện" đều mang nhãn `thap` —
tín hiệu hoàn toàn nhất quán, không có nhiễu nhãn. Vậy lỗi không đến từ dữ liệu sai mà từ
việc model **chưa học hết** tín hiệu này sau 30 step: nó học được intent/product/sentiment
(các trường gắn với từ khoá rõ ràng như "Bị lỗi", "Hoàn tiền"), nhưng với urgency nó vẫn
nghiêng về giá trị "an toàn" ở giữa. Ba ca này cũng đều có câu mô tả vấn đề (chưa nhận
tiền, thiếu phụ kiện, bị lỗi) — nội dung *nghe* có vẻ gấp — trong khi khách lại nói "khi
nào tiện"; model đặt nặng nội dung vấn đề hơn cụm chỉ mức độ gấp. Ngược lại, các ca thắng
có cụm urgency rõ hơn ("Sớm nhé", "Mong shop phản hồi") hoặc không xung đột với nội dung.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không** deploy bản fine-tune này ở dạng hiện tại, dù nó đạt 0.970
trên tác vụ chính. Lý do là cổng hồi quy: nó đổi 0.205 điểm target lấy 0.269 điểm năng
lực chung, và một model CSKH thật sẽ nhận cả những câu không phải ticket. Nếu sản phẩm là
một bộ phân loại *chỉ* nhận ticket (đầu vào đã được lọc trước), bản fine-tune rất đáng
dùng: chính xác hơn prompt tối ưu 0.205 với prompt ngắn hơn nhiều. Nhưng phán quyết lab
đo đúng rủi ro thật, nên tôi chấp nhận FAILED thay vì tìm cách lách. Hướng đi tiếp theo là
trộn 1–5 % dữ liệu phổ thông vào tập train và đo lại cả hai nhóm.

Về đòn bẩy: thí nghiệm cho một thứ tự rất rõ. **Learning rate** là đòn bẩy mạnh nhất —
sai một bậc độ lớn đưa target từ 0.970 về 0.000, vì adapter quá yếu để thay đổi token đầu
tiên của câu trả lời. **Dữ liệu** là đòn bẩy thứ hai, và là thứ quyết định phán quyết:
dữ liệu một dạng duy nhất sinh ra quên thảm hoạ, không cấu hình LoRA nào sửa được điều đó.
**Mask** là điều kiện cần — `supervised_fraction = 0.41` và hai assert xanh nên mọi số
phía sau mới có nghĩa. **Vị trí adapter và rank** gần như không quan trọng trên tác vụ
này (hoà 0.970), và **lượng tử hoá 4-bit** chỉ tốn 0.03 điểm để tiết kiệm 56 % VRAM. Cuối
cùng, train loss đã đánh lừa hai lần: nó xếp `attn_only` trên `correct` khi thực tế hai
cái hoà, và nó cho `wrong_lr` một đường cong "đang học" khi thực tế điểm tác vụ bằng 0.

**Ba điều tôi học được:**

1. **Train loss và token accuracy có thể nói dối hoàn toàn.** `wrong_lr` có loss giảm đều
   và `mean_token_accuracy` ~0.79, nhưng không sinh nổi một JSON hợp lệ (format 0.000).
   Teacher forcing che mất việc model đi sai ngay từ token đầu tiên. Từ giờ tôi luôn đo
   bằng *generation* trên tập eval, không dừng ở loss.
2. **Tác vụ dễ làm thí nghiệm mất khả năng phân biệt.** `correct` và `attn_only` cùng
   0.970 vì cả hai chạm trần, nên tôi không kiểm chứng được câu "vị trí thắng rank" của
   deck — chỉ bác bỏ được "rank là đòn bẩy". Muốn so cấu hình, tác vụ phải đủ khó để điểm
   chưa bão hoà.
3. **Kiểm tra cấu hình chạy thật, không tin form nhập liệu.** Lần chạy đầu của tôi vẫn ở
   `EVAL_LIMIT=8` dù tôi đã đổi ô chọn, và bảng kết quả trông hoàn toàn bình thường — chỉ
   cột `n = 8` để lộ ra. Nếu không đọc cột đó, tôi đã viết report trên 8 mẫu. Giờ tôi luôn
   đọc dòng `eval_limit=full` và cột `n` trước khi tin bất kỳ con số nào.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) trộn ~3 % mẫu hỏi–đáp phổ thông (replay) vào
225 mẫu train, train lại `correct` và xem regression có quay về trong ngưỡng 0.02 mà target
vẫn trên 0.765 không — đó là phép thử trực tiếp cho chẩn đoán ở §5; (2) chạy NB6 để merge
adapter và đo lại latency, vì bản fine-tune hiện chậm hơn (b) (1604 vs 1154 ms) dù prompt
ngắn hơn — tôi nghi chi phí đến từ adapter chưa merge, và `attn_only` (ít module hơn)
nhanh hơn hẳn là một dấu hiệu ủng hộ giả thuyết đó.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
