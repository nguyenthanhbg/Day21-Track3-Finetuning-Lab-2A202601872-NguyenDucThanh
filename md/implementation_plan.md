# Kế Hoạch Thực Hiện Lab 21 - Fine-tuning LLMs

Kế hoạch này sẽ hướng dẫn chúng ta đi qua từng bước của Lab 21, từ cài đặt môi trường cho đến khi tạo ra file báo cáo nộp bài. Vì quá trình huấn luyện AI mất khá nhiều thời gian và yêu cầu phần cứng cụ thể, chúng ta sẽ cần phối hợp cùng nhau.

## ⚠️ User Review Required (Cần Bạn Xác Nhận)

> [!IMPORTANT]
> **Vui lòng trả lời phần "Open Questions" bên dưới trước khi chúng ta bắt đầu bước đầu tiên!** Việc này quyết định chúng ta sẽ chạy mã lệnh (code) ở đâu để không bị lỗi bộ nhớ.

## ❓ Open Questions (Câu hỏi dành cho bạn)

1. **Cấu hình máy tính hiện tại của bạn:** Máy tính bạn đang dùng có Card màn hình rời (GPU) không? (Ví dụ: NVIDIA RTX 3060, 4060, hay các dòng cao hơn). Nếu có, bộ nhớ (VRAM) của nó là bao nhiêu GB?
2. **Nơi huấn luyện mong muốn:** Bạn muốn tôi (AI) tự động chạy lệnh và huấn luyện trực tiếp trên máy tính này của bạn (chúng ta sẽ cài `COMPUTE_TIER=LAPTOP` hoặc `T4` nếu máy bạn đủ mạnh), **HAY** bạn muốn dùng máy tính này để chuẩn bị dữ liệu (Notebook 1), sau đó bạn sẽ tải bộ source này lên Google Colab để huấn luyện?

---

## 🛠️ Các Bước Thực Hiện (Proposed Changes & Execution)

Chúng ta sẽ chia dự án thành 3 giai đoạn chính.

### Giai đoạn 1: Chuẩn bị và Nghiệm thu đầu vào (Làm ngay trên máy này)

Ở bước này, chúng ta không cần đến GPU xịn. Mục tiêu là cài đặt môi trường Python, kiểm tra source code và hoàn thành Notebook 1 (`NB1`).

- **Bước 1.1:** Cài đặt các thư viện Python cơ bản. Tôi sẽ chạy lệnh `make setup-cpu` trên máy của bạn. (Nếu bạn xác nhận máy bạn có GPU NVIDIA mạnh, tôi sẽ đổi lệnh thành `make setup`).
- **Bước 1.2:** Chạy kiểm tra nhanh toàn bộ source code bằng lệnh `make smoke`. Nếu tất cả unit test màu xanh (PASS), source code của bạn an toàn.
- **Bước 1.3:** Chạy **Notebook 1** (Lệnh `make nb1`).
  - *Mục tiêu:* Kiểm tra Chat Template, trích xuất con số `max_length` từ phân phối p95, và quan trọng nhất là chứng minh **Loss Mask** hoạt động đúng (chỉ phạt mô hình khi nó sinh ra câu trả lời sai, không phạt câu hỏi).
  - *Đầu ra:* Các file json như `mask_proof.json`, `token_stats.json` sinh ra trong thư mục `results/`.

### Giai đoạn 2: Cấu hình và Huấn luyện (Core Training)

Đây là giai đoạn ngốn nhiều tài nguyên. Tùy thuộc vào câu trả lời của bạn ở trên, tôi sẽ tự động chạy các lệnh sau trên máy bạn, hoặc hướng dẫn bạn bấm nút trên Colab.

- **Bước 2.1 - Chạy Notebook 2 (`make nb2`):** Đóng băng bộ đánh giá. Đo điểm số của Qwen3.5 gốc trên tập dữ liệu đánh giá để làm mốc (Baseline) trước khi train.
- **Bước 2.2 - Chạy Notebook 3 (`make nb3`):** Cấu hình LoRA (rank, alpha, modules) theo chuẩn "vùng-không-hối-tiếc" và tiến hành huấn luyện. Bản này sẽ sinh ra thư mục `adapters/correct/`.
- **Bước 2.3 - Chạy Notebook 4 (`make nb4`):** Huấn luyện 3 bản sao cố tình làm sai cấu hình để lấy dữ liệu đối chứng (Đóng vai trò quan trọng nhất trong báo cáo). Sẽ tiêu tốn thêm khoảng 45-60 phút.

### Giai đoạn 3: Phán quyết và Đóng gói nộp bài

Sau khi đã có đủ bộ weights của 4 lần huấn luyện.

- **Bước 3.1 - Chạy Notebook 5 (`make nb5`):** Chấm điểm 4 mô hình trên 4 tiêu chí (target, regression, format, latency) và ra quyết định (Verdict). 
- **Bước 3.2 - Điền Report:** Tôi sẽ thu thập toàn bộ các số liệu sinh ra trong thư mục `results/`, tự động phân tích và viết nháp file `submission/REPORT.md` cho bạn (đảm bảo viết đủ 150 từ kết luận, nhặt đủ các ca model thắng/thua). Bạn sẽ là người review lại lời lẽ.
- **Bước 3.3 - Gatekeeper (`make verify`):** Tôi sẽ chạy lệnh này để chắc chắn bạn không vi phạm bất kỳ lỗi liêm chính nào trước khi nén file.

---

## 🔍 Verification Plan (Kế hoạch kiểm tra)

1. **Automated Tests:** Xuyên suốt quá trình, lệnh `make smoke` sẽ được dùng để đảm bảo pipeline không vỡ. Lệnh `make verify` ở cuối cùng sẽ đóng vai trò trọng tài tối cao.
2. **Manual Verification:** Bạn và tôi sẽ phải mở file `results/mask_proof.json` để tận mắt kiểm chứng xem chữ `true` và `false` của mask có đang đặt đúng vị trí câu hỏi/câu trả lời hay không. Đừng chỉ tin vào mã code!
