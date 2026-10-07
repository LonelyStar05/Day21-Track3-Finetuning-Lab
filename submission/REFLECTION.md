# Reflection — Lab 21

**Họ tên**: Nguyễn Tú Tài  **MSSV**: 2A202602455

**1. Điều gì làm bạn ngạc nhiên nhất?**

Run `wrong_lr`. Loss giảm đều từ 2.16 xuống 1.12, token accuracy lên khoảng 0.79, nhìn log
mình nghĩ nó chỉ học chậm hơn chút thôi. Đem đi chấm thì 0 điểm, không ra được cái JSON nào,
y hệt model gốc chưa train. Chỉ đổi learning rate từ 1e-4 xuống 1e-5 mà mất sạch.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ mình nghĩ. Mình tưởng khó nhất là chỉnh LoRA, nhưng thật ra mất thời gian
nhất là chạy cho đúng cấu hình. Lần đầu mình đã chọn EVAL_LIMIT trống rồi mà nó vẫn chạy
8 mẫu, phải nhìn cột `n = 8` mới biết. Phải chạy lại NB2 với NB5 thêm khoảng 18 phút. May
là adapter đã lưu nên không phải train lại từ đầu.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước mình nghĩ fine-tune xong mà điểm tác vụ cao là dùng được. Bản của mình ra 0.970, hơn
prompt tối ưu 0.205, nhưng vẫn FAILED vì hỏi câu phổ thông thì điểm tụt 0.269. Điểm cao ở
một việc không nói lên model đã quên mất những gì.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Mình dùng Claude để hướng dẫn chạy Colab, đọc kết quả và soạn report. Nó không tự chạy lab
được vì môi trường của nó bị chặn HuggingFace và không có GPU, nên mình phải tự chạy trên
Colab rồi gửi số cho nó. Lúc viết script tạo report, ban đầu nó gắn nhãn "FT thắng/thua"
theo điểm của riêng bản fine-tune chứ không so với baseline (b), sai ý của rubric, sau đó
mới sửa lại.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Chốt tập eval và đo prompt tốt nhất trước, kèm thêm một bộ câu hỏi ngoài tác vụ để biết
model có quên gì không. Rồi ngay từ đầu trộn 1–5 % dữ liệu phổ thông vào tập train, vì lab
này cho thấy dữ liệu chỉ có một kiểu mới là thứ làm hỏng kết quả, chứ không phải cấu hình LoRA.
