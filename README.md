# Neural Machine Translation (English → Vietnamese) with Transformer

Dự án này tập trung vào việc nghiên cứu và triển khai các hệ thống dịch máy cho cặp ngôn ngữ Anh-Việt, bao gồm việc xây dựng mô hình Transformer từ các thành phần cơ bản và tinh chỉnh mô hình pre-trained cho lĩnh vực chuyên biệt. Đây là nội dung thuộc bài tập lớn môn Xử lý ngôn ngữ tự nhiên năm 2025.



## 📌 Tổng quan dự án

Dự án triển khai hai hướng tiếp cận chính:
1.  **Transformer Code from Scratch**: Tự xây dựng kiến trúc Transformer Seq2Seq và huấn luyện trên bộ dữ liệu **IWSLT2015 En-Vi**.
2.  **Fine-tuning MarianMT**: Sử dụng mô hình `Helsinki-NLP/opus-mt-en-vi` và tinh chỉnh cho bài toán dịch thuật chuyên ngành y tế trong khuôn khổ cuộc thi **VLSP Medical MT**.

## 🏗️ Kiến trúc & Tính năng

### 1. Transformer từ đầu (Scratch)
Triển khai đầy đủ các thành phần cốt lõi bằng PyTorch:
* **Multi-Head Attention**: Cơ chế chú ý đa đầu giúp mô hình hiểu ngữ cảnh tốt hơn.
* **Encoder/Decoder Blocks**: Các khối mã hóa và giải mã tiêu chuẩn với Residual Connection và Layer Normalization.
* **Positional Encoding**: Nhúng thông tin vị trí vào chuỗi đầu vào do Transformer không có tính tuần tự như RNN.
* **Label Smoothing & Noam Scheduler**: Tối ưu hóa quá trình hội tụ của mô hình.
* **Beam Search Decoding**: Thuật toán giải mã giúp tìm ra bản dịch có xác suất cao nhất.
* **Tokenization**: Sử dụng Byte Pair Encoding (BPE) để xử lý từ vựng hiệu quả.

### 2. Fine-tuning cho Y tế (VLSP Medical MT)
* Tinh chỉnh mô hình **MarianMT** trên tập dữ liệu chuyên ngành y khoa.
* Khả năng dịch chính xác các thuật ngữ y học phức tạp mà các mô hình thông thường dễ mắc lỗi.

## 💾 Model Checkpoints

Bạn có thể tải xuống các trọng số mô hình (weights) đã được huấn luyện sẵn tại liên kết dưới đây để chạy thử nghiệm ngay mà không cần huấn luyện lại:

👉 **[Google Drive - Trained Models](https://drive.google.com/drive/folders/1gDUzKpvDsgoGJeulh3416IfbyYA_o9Qy?usp=drive_link)**

## 📊 Kết quả thực nghiệm

Kết quả được đo lường bằng chỉ số BLEU (Bilingual Evaluation Understudy):

| Phương pháp | Tập dữ liệu | Số câu Test | BLEU Score |
| :--- | :--- | :--- | :--- |
| **Transformer (Scratch)** | IWSLT2015 | 500 | **13.18** |
| **MarianMT (Fine-tune)** | VLSP Medical | 3000 | **47.49** |

## 📁 Cấu trúc thư mục

* `transformer.ipynb`: Notebook chi tiết quá trình xây dựng mô hình từ đầu, từ khâu xử lý dữ liệu đến lúc inference.
* `NLP_task2.ipynb`: Notebook thực hiện fine-tuning mô hình MarianMT trên GPU (A100/T4).
* `NLP_report_v2.pdf`: Báo cáo kỹ thuật chi tiết về lý thuyết và phân tích kết quả.
