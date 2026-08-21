# Hướng Dẫn Chi Tiết Lab 21 — Fine-tuning LLMs (Track 3)

## 1. Tổng Quan và Ý Nghĩa Của Bài Lab
**Ý nghĩa:** Bài lab này giúp bạn thực hành quá trình fine-tune một mô hình ngôn ngữ lớn (LLM) mã nguồn mở bằng phương pháp LoRA/QLoRA. Điểm cốt lõi không chỉ là việc huấn luyện mô hình thành công, mà là **chứng minh** được bản fine-tune của bạn thực sự tốt hơn so với mô hình gốc (base model) khi mô hình gốc đã được cung cấp một prompt tối ưu.

**Mục tiêu chính (2 câu hỏi bắt buộc phải trả lời được):**
1. Phần văn bản được mô hình tính loss có thực sự chỉ là câu trả lời không? (Phải giải mã ngược để kiểm chứng mask).
2. Bản fine-tune có thắng được base model với prompt tối ưu không? Nếu không thắng, bạn có phát hiện ra được điều đó thông qua dữ liệu đánh giá hay không? Điểm số của bạn không nằm ở chỗ fine-tune thắng, mà nằm ở chỗ bạn **biết** nó có thắng hay không.

## 2. Bài Toán Đang Dạy Model
**Đầu vào:** Ticket CSKH tiếng Việt. Ví dụ: *"Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi."*
**Đầu ra bắt buộc:** Đúng MỘT chuẩn JSON, đúng 4 khóa, **không bọc markdown, không giải thích dài dòng**:
```json
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}
```
**Các tập giá trị cho phép:**
- `intent`: `doi_tra` | `van_chuyen` | `hoan_tien` | `san_pham_loi` | `hoi_thong_tin`
- `urgency`: `cao` | `trung_binh` | `thap`
- `sentiment`: `tieu_cuc` | `trung_tinh` | `tich_cuc`
- `product`: Tên sản phẩm xuất hiện trong ticket (so khớp sau khi bỏ dấu).

## 3. Bạn Cần Chuẩn Bị Những Gì?
- **Tài khoản Google:** Dùng để chạy Google Colab (Môi trường khuyến nghị).
- **GPU:** Colab Free T4 (khoảng 14,6 GB khả dụng) hoặc GPU cá nhân (Laptop RTX 3060/4060, PC có VRAM lớn). 
- **Thời gian:** Khoảng 95-110 phút core cho (NB1 → NB5) trên GPU T4. Lần chạy đầu sẽ mất thêm chút thời gian tải ~9.32 GB trọng số mô hình. Đừng bắt đầu train nếu Colab sắp hết giờ!
- **Kiến thức nền:** Hiểu về transformer, token, GPU VRAM. Không cần thiết phải có kinh nghiệm fine-tune từ trước. KHÔNG CẦN chuẩn bị dataset riêng trừ khi muốn làm điểm thưởng.

## 4. Tiêu Chí Hoàn Thành (Khi Nào Được Coi Là Xong?)
Bạn hoàn thành bài lab khi **đồng thời** có đủ các bằng chứng sau (không phải chỉ là "loss đã giảm"):

1. **Mask đúng bằng mắt:** File `results/mask_proof.json` phải hiển thị `answer_is_supervised: true`, `question_is_masked: true`, và `supervised_fraction < 0.95`. Bạn cần chép 3-5 dòng đầu của đoạn tính loss vào file `REPORT.md`.
2. **Mốc đánh giá bị đóng băng:** File `results/baselines_frozen.json` có điểm `(a)` và `(b)`, trong đó `(b).target > (a).target`. **Tuyệt đối không sửa tập eval sau đó.**
3. **Có bản LoRA chuẩn:** Thư mục `adapters/correct/` chứa `adapter_model.safetensors` và config. File `results/runs.csv` có dòng ghi nhận bản `correct` kèm loss, VRAM, và `max_steps`.
4. **Ba đối chứng công bằng:** Cả 3 bản đối chứng phải có cùng `max_steps` với bản `correct`. Bản `attn_only` lệch ngân sách tham số `< 5%`. Mỗi bản đối chứng chỉ được lệch **một biến duy nhất** (vị trí / LR / 4-bit).
5. **Phán quyết 4 nhóm:** File `results/verdict.json` ghi nhận PASSED/FAILED. File `results/autopsy.json` xếp hạng 4 runs dựa trên điểm **target**, tuyệt đối không xếp hạng bằng `final_loss`. File `results/qualitative.json` phải có đủ cả ca model thắng lẫn ca model **thua**.
6. **Báo cáo khớp số:** File `submission/REPORT.md` điền đủ 7 mục, xóa hết các placeholder `<điền>`, mọi con số phải lấy đúng từ thư mục `results/`. Đoạn kết luận dài ít nhất 150 từ. Phải liệt kê ít nhất 2 ví dụ bản fine-tune bị thua.
7. **Xác minh qua script:** Chạy lệnh `make verify` (hoặc `python scripts/verify.py`) phải trả về mã thoát là 0. Bất kỳ dòng FAIL nào cũng phải sửa. Bất kỳ dòng WARN nào cũng phải được giải thích trong báo cáo.

## 5. Chọn Tier Phần Cứng (`COMPUTE_TIER`)
Sửa `COMPUTE_TIER` trong file `.env` theo máy của bạn:
- `CPU`: Mô hình `Qwen3.5-0.8B`. Dùng để chạy thử NB1 + toàn bộ test. Cực nhanh để test luồng nhưng không dùng để train.
- `LAPTOP`: Mô hình `Qwen3.5-2B`. Phù hợp Laptop GPU 8-12 GB.
- **`T4` (Mặc định)**: Mô hình `Qwen3.5-4B`. Dùng Colab Free T4 16 GB. (Đây là đường chuẩn).
- `BIGGPU`: Mô hình `Qwen3.5-9B`. Cho các cỗ máy >= 22 GB VRAM.

## 6. Các Bước Thực Hiện Cụ Thể (Các Notebook)
* **NB1 (`01_data_and_mask.py` - Chạy CPU):** Dọn dữ liệu, kiểm tra chat template, chứng minh mask.
* **NB2 (`02_baselines.py` - Cần GPU):** Đo hiệu suất base model với 2 dạng prompt (thường & tối ưu). 
* **NB3 (`03_train_correct.py` - Cần GPU):** Huấn luyện phiên bản LoRA đúng cấu hình.
* **NB4 (`04_misconfig_autopsy.py` - Cần GPU):** Huấn luyện 3 phiên bản sai cấu hình (đối chứng) để mổ xẻ so sánh.
* **NB5 (`05_evaluate_and_verdict.py`):** Đánh giá tất cả, xuất ra phán quyết và ca bệnh định tính.
* **NB6 (Tùy chọn):** Merge adapter lấy điểm thưởng.

## 7. Mẹo Chạy Rút Gọn (Khi Gặp Áp Lực Thời Gian)
Lưu ý: Chỉ dùng để test hoặc khi hết giờ, lúc nộp bài nếu rút gọn sẽ bị ghi vào log `results/`.
- `EVAL_LIMIT=8 make pipeline`: Chỉ đánh giá trên ít mẫu (phần SINH văn bản sẽ ngắn lại).
- `EPOCHS=1 make pipeline`: Giảm một nửa số step huấn luyện. (Nếu dùng thì lệnh này tự áp dụng cho cả NB3 và NB4 để đảm bảo nguyên tắc đối chứng công bằng).

## 8. Những Điều CẤM KỴ (Quy tắc liêm chính)
1. **Không sửa tập eval:** Cấm đổi tập đánh giá một khi đã ra kết quả (công cụ chấm đối chiếu checksum).
2. **Không làm yếu prompt (b):** Không được sửa prompt tối ưu thành prompt tệ hại đi để gian lận, làm cho bản fine-tune của bạn tự nhiên trông có vẻ "chiến thắng" áp đảo.
3. **Mọi file báo cáo phải là SỰ THẬT:** Tuyệt đối không "chế" số cho đẹp. Báo cáo phải khớp 100% số sinh ra trong `results/`. 
