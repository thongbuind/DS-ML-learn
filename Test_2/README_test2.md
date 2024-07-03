## Kết quả Đánh giá Mô hình

| Class         | Precision | Recall | F1-Score | Support |
|:-------------:|:---------:|:------:|:--------:|:-------:|
| Lost          |   0.76    |  0.81  |   0.78   |   129   |
| Tie           |   0.25    |  0.06  |   0.10   |   16    |
| Win           |   0.70    |  0.72  |   0.71   |   82    |
| **Accuracy**  |           |        |   0.73   |   227   |
| **Macro avg** |   0.57    |  0.53  |   0.53   |   227   |
| **Weighted avg**| 0.70    |  0.73  |   0.71   |   227   |

### Chú thích:
- **Precision**: Tỷ lệ mẫu dương tính được dự đoán đúng trên tổng số mẫu dương tính được dự đoán.
- **Recall**: Tỷ lệ mẫu dương tính được dự đoán đúng trên tổng số mẫu dương tính thực tế.
- **F1-Score**: Trung bình hài hòa của Precision và Recall.
- **Support**: Số lượng mẫu thực tế thuộc mỗi lớp.
- **Accuracy**: Tỷ lệ mẫu được dự đoán đúng trên tổng số mẫu.
- **Macro avg**: Trung bình của Precision, Recall và F1-Score tính trên các lớp, không có trọng số.
- **Weighted avg**: Trung bình của Precision, Recall và F1-Score tính trên các lớp, có trọng số theo số lượng mẫu của từng lớp.