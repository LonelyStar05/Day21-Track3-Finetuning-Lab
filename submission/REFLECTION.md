# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

`wrong_lr`. Loss của nó giảm đều từ 2.16 xuống 1.12 và token accuracy lên ~0.79 — nhìn
log thì tưởng chỉ học chậm hơn. Nhưng trên tập eval nó ra target 0.000, format 0.000:
không một JSON hợp lệ nào, giống hệt base model chưa train. Chỉ đổi LR từ 1e-4 xuống
1e-5 mà mất toàn bộ tác vụ.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở train, mà ở việc chạy đúng cấu hình. Lần chạy đầu vẫn ở `EVAL_LIMIT=8` dù
tôi đã chọn ô trống trên form Colab, và tôi chỉ phát hiện ra nhờ cột `n = 8` trong bảng.
Phải chạy lại NB2 + NB5 (~18 phút). May là adapter đã lưu, nên không phải train lại.
Tôi đã dự đoán phần khó sẽ là cấu hình LoRA.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ fine-tune đạt điểm cao trên tác vụ là đủ để dùng. Bản của tôi đạt 0.970 (hơn
prompt tối ưu 0.205) nhưng vẫn FAILED vì mất 0.269 điểm ở câu hỏi phổ thông. Điểm tác vụ
cao không nói gì về những thứ model đã quên.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Dùng để hướng dẫn chạy Colab, đọc kết quả và soạn report. Nó không chạy được lab trong môi
trường của nó (mạng chặn HuggingFace, không có GPU), nên tôi phải tự chạy trên Colab. Khi
soạn script tạo report, ban đầu nó gắn nhãn "FT thắng/thua" theo điểm tuyệt đối của bản
fine-tune thay vì so với baseline (b) — sai ý nghĩa của rubric 3.4, sau đó đã sửa.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng tập eval và đo baseline prompt tối ưu trước, kèm một tập câu hỏi ngoài tác vụ
để đo hồi quy. Sau đó ngay từ đầu trộn 1–5 % dữ liệu phổ thông vào tập train, vì lab này
cho thấy dữ liệu một dạng duy nhất là thứ làm hỏng phán quyết, không phải cấu hình LoRA.
