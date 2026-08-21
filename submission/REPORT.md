# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đức Thành  **MSSV**: 2A202601872  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 14.6 GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 250 / 50 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 30 |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? (Không áp dụng)

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3167.5 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1028.6 |
| (c) LoRA fine-tune | 0.970 | 0.6556 | 1.000 | 1443.0 |

**(b) có thật sự mạnh hơn (a) không?** `có` — (từ 0.0 lên 0.765). 
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? Không sửa, vì hiệu năng của prompt gốc (b) đã phân loại rất tốt với target đạt 0.765.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6281 | 0.970 | 971.9 | 12.01 |
| `attn_only` | q,v | *(matched)* | 32,456,704 | 0.0001 | 0.5371 | 0.970 | 836.2 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | 0.000 | 979.8 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 1036.5 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` hoà với `correct` (cùng đạt 0.97). Tuy nhiên thứ tự này không hoàn toàn giống với train loss, khi mà train loss của `attn_only` lại thấp hơn hẳn (`0.5371` so với `0.6281`). Điều này nói lên rằng khi số lượng tham số huấn luyện (rank) được đẩy lên để ngang bằng nhau, vị trí đặt adapter không còn là giới hạn sống còn nữa, vì mô hình vẫn có đủ dung lượng (capacity) để ghi nhớ task.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
`wrong_lr` cho đường loss rất cao (`1.5704`) và target thê thảm rớt về `0.0`. Nếu chỉ nhìn loss cao mà không biết do cấu hình sai LR, ta rất dễ kết luận sai lầm rằng model này quá nhỏ không đủ khả năng học, hoặc tập dữ liệu quá khó. Từ đó dẫn đến cách chữa cháy sai lầm như việc tăng parameter hay thu thập thêm dữ liệu, trong khi sự thật chỉ là LR của LoRA cần lớn hơn nhiều so với full fine-tuning.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` giúp tiết kiệm được một lượng lớn VRAM, giảm gần 5GB (từ `12.01 GB` xuống còn `7.09 GB`). Cái giá phải trả là thời gian huấn luyện lâu hơn một chút (1036s so với 971s) và điểm target giảm nhẹ (từ 0.97 xuống 0.94). Dựa vào thực tế số đo này, em không hoàn toàn ủng hộ khuyến nghị "không dùng QLoRA" của hãng, bởi việc đánh đổi 3% độ chính xác để train được trên các máy tính cấu hình yếu là vô cùng xứng đáng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.102` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
Bản Fine-tune đã tăng cường khả năng phân loại JSON một cách cực kỳ xuất sắc (tăng 20.5% so với prompt tối ưu). Tuy nhiên, cổng hồi quy đã bị FAILED do sự sụt giảm nặng nề trên tập dữ liệu tổng quát (giảm tới -0.102, vượt xa mức dung sai 0.02). Điều này xảy ra bởi hội chứng Catastrophic Forgetting (quên thảm họa), khi ta ép mô hình chỉ sinh ra đúng một kiểu format JSON duy nhất mà không cho nó tập luyện với các câu hỏi tự nhiên. Điều đó nói lên rằng với các bài toán áp đặt định dạng gắt gao như thế này, chúng ta không thể chỉ huấn luyện model trên 100% dữ liệu đích mà bắt buộc phải có bước hòa trộn thêm 1-5% dữ liệu replay (các câu hỏi chatbot tổng quát) để giữ lại "cá tính" linh hoạt vốn có của mô hình gốc.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232... | doi_tra, cao | JSON sai | JSON đúng chuẩn | ✅ FT thắng |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931... | hoan_tien, trung_binh | JSON rác | JSON đúng chuẩn | ✅ FT thắng |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124... | hoan_tien, trung_binh | JSON đúng | Cắt đuôi | ❌ **FT thua** |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548... | san_pham_loi, trung_binh | JSON đúng | Cắt đuôi | ❌ **FT thua** |

Có mẫu chung nào ở các ca FT thua không?
Điểm chung của hầu hết các ca thua là điểm số `ft_score = 0.75` thay vì 1.0 hoặc 0.0. Lý do là vì ở các ca thua này, chuỗi JSON mà model sinh ra bị cắt cụt giữa chừng ngay tại vị trí xuất hiện key `"sentiment": `. Điều này chứng tỏ vấn đề không nằm ở khả năng phân loại của Fine-tune, mà nằm ở giới hạn về độ dài sinh token (`max_new_tokens`) hoặc điều kiện ngắt (stop token) trong hàm đánh giá đang quá chặt, khiến câu bị ngắt khi nó chưa kịp đóng ngoặc JSON.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao?
Nếu triển khai bản fine-tune này như một chatbot giao tiếp trực tiếp với khách hàng, câu trả lời chắc chắn là KHÔNG, vì mô hình đã FAILED cổng hồi quy, mất đi khả năng nói chuyện bình thường do bị quên thảm họa. Tuy nhiên, nếu triển khai dưới dạng một hệ thống API backend "ẩn" chỉ nhận nhiệm vụ chuyên biệt là trích xuất và phân loại Ticket ra file JSON để lưu vào Database, thì mô hình này CÓ THỂ deploy được vì nó hoàn thành cực kì xuất sắc với target đạt 0.97, cộng thêm tốc độ latency phản hồi đã giảm xuống đáng kể so với việc bắt một mô hình chưa train thực hiện. Đòn bẩy thực sự lớn nhất trong lab này chính là Learning Rate và Prompting: một LR sai có thể hủy diệt mọi thứ, còn một Prompt tối ưu tốt (như (b)) đôi khi đã giải quyết được tới gần 80% bài toán mà chưa cần phải tốn công Fine-tune tốn kém.

**Ba điều tôi học được** (cụ thể, không generic):
1. Learning rate của thuật toán LoRA cần được cấu hình lớn hơn (thường là 10 lần) so với Full Fine-tuning, nếu dùng chung LR thì model gần như không học được gì.
2. Vị trí đặt adapter (rank placement) không phải lúc nào cũng quyết định thắng thua, bởi nếu ta dùng rank để cân bằng lại số lượng parameter, `attn_only` hoàn toàn có thể mạnh ngang ngửa với `all-linear`.
3. Phải luôn xây dựng một tập kiểm duyệt hồi quy (Regression Test) khi Fine-tune, vì mô hình rất dễ bị cuốn theo task mới mà vứt bỏ toàn bộ tri thức nền tảng trước đó.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn thêm 5% dữ liệu hội thoại thông thường (Replay Data) vào tập train của bài lab để huấn luyện lại, qua đó xem có giải cứu được điểm số của cổng hồi quy và lật ngược tình thế từ FAILED sang PASSED hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
