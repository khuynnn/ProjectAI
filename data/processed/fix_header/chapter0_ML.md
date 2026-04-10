Vũ Hữu Tiệp

# Machine Learning cơ bản

Machine Learning cơ bản

Cập nhật lần cuối: 02/01/2022.

Bản quyền ©2016 – 2020: Vũ Hữu Tiệp

Mọi hình thức sao chép, in ấn đều cần được sự đồng ý của tác giả. Mọi chia sẻ đều cần được dẫn nguồn tới <https://github.com/tiepvupsu/ebookMLCB>.

Mục lục

| 0 | Lời nói đầu<br> |                              | 15 |
|---|-----------------|------------------------------|----|
|   | 0.1             | Mục đích của cuốn sách<br>   | 16 |
|   | 0.2             | Hướng tiếp cận của cuốn sách | 17 |
|   | 0.3             | Đối tượng của cuốn sách      | 17 |
|   | 0.4             | Yêu cầu về kiến thức         | 18 |
|   | 0.5             | Mã nguồn đi kèm              | 19 |
|   | 0.6             | Bố cục của cuốn sách<br>     | 19 |
|   | 0.7             | Các lưu ý về ký hiệu         | 19 |
|   | 0.8             | Tham khảo thêm               | 20 |
|   | 0.9             | Đóng góp ý kiến              | 21 |
|   | 0.10            | Lời cảm ơn<br>               | 21 |
|   | 0.11            | Bảng các ký hiệu             | 21 |
|   |                 |                              |    |
|   |                 | Phần I Kiến thức toán cơ bản |    |
|   |                 |                              |    |
| 1 |                 | Ôn tập Đại số tuyến tính<br> | 24 |
|   | 1.1             | Lưu ý về ký hiệu<br>         | 24 |
|   | 1.2             | Chuyển vị và Hermitian       | 24 |

|   | 1.3  | Phép nhân hai ma trận                         | 25 |
|---|------|-----------------------------------------------|----|
|   | 1.4  | Ma trận đơn vị và ma trận nghịch đảo          | 26 |
|   | 1.5  | Một vài ma trận đặc biệt khác<br>             | 28 |
|   | 1.6  | Định thức                                     | 29 |
|   | 1.7  | Tổ hợp tuyến tính, không gian sinh<br>        | 30 |
|   | 1.8  | Hạng của ma trận                              | 32 |
|   | 1.9  | Hệ trực chuẩn, ma trận trực giao              | 33 |
|   | 1.10 | Biễu diễn vector trong các hệ cơ sở khác nhau | 34 |
|   | 1.11 | Trị riêng và vector riêng                     | 35 |
|   | 1.12 | Chéo hoá ma trận<br>                          | 36 |
|   | 1.13 | Ma trận xác định dương                        | 37 |
|   | 1.14 | Chuẩn<br>                                     | 40 |
|   | 1.15 | Vết                                           | 42 |
| 2 |      | Giải tích ma trận<br>                         | 43 |
|   | 2.1  | Gradient của hàm trả về một số vô hướng<br>   | 43 |
|   | 2.2  | Gradient của hàm trả về vector<br>            | 45 |
|   | 2.3  | Tính chất quan trọng của gradient             | 46 |
|   | 2.4  | Gradient của các hàm số thường gặp<br>        | 46 |
|   | 2.5  | Bảng các gradient thường gặp                  | 49 |
|   | 2.6  | Kiểm tra gradient<br>                         | 49 |
| 3 |      | Ôn tập Xác suất<br>                           | 54 |
|   | 3.1  | Xác suất<br>                                  | 54 |
|   | 3.2  | Một vài phân phối thường gặp<br>              | 62 |

| 4 |     | Ước lượng tham số mô hình<br>                       | 67  |
|---|-----|-----------------------------------------------------|-----|
|   | 4.1 | Giới thiệu<br>                                      | 67  |
|   | 4.2 | Ước lượng hợp lý cực đại<br>                        | 67  |
|   | 4.3 | Ước lượng hậu nghiệm cực đại                        | 73  |
|   | 4.4 | Tóm tắt                                             | 77  |
|   |     | Phần II Tổng quan                                   |     |
| 5 |     | Các khái niệm cơ bản                                | 80  |
|   | 5.1 | Nhiệm vụ, kinh nghiệm, phép đánh giá                | 80  |
|   | 5.2 | Dữ liệu                                             | 81  |
|   | 5.3 | Các bài toán cơ bản trong machine learning          | 82  |
|   | 5.4 | Phân nhóm các thuật toán machine learning<br>       | 84  |
|   | 5.5 | Hàm mất mát và tham số mô hình                      | 86  |
| 6 |     | Các kỹ thuật xây dựng đặc trưng<br>                 | 88  |
|   | 6.1 | Giới thiệu<br>                                      | 88  |
|   | 6.2 | Mô hình chung cho các bài toán machine learning<br> | 89  |
|   | 6.3 | Một số kỹ thuật trích chọn đặc trưng<br>            | 91  |
|   | 6.4 | Học chuyển tiếp cho bài toán phân loại ảnh          | 96  |
|   | 6.5 | Chuẩn hoá vector đặc trưng                          | 99  |
| 7 |     | Hồi quy tuyến tính<br>100                           |     |
|   | 7.1 | Giới thiệu<br>                                      | 100 |
|   | 7.2 | Xây dựng và tối ưu hàm mất mát                      | 101 |
|   | 7.3 | Ví dụ trên Python                                   | 103 |

|    |            | ·                             |
|----|------------|-------------------------------|
|    | 7.4        | Thảo luận 106                 |
| 8  | Quá        | <b>á khớp</b>                 |
|    | 8.1        | Giới thiệu                    |
|    | 8.2        | Xác thực                      |
|    | 8.3        | Cơ chế kiểm soát              |
|    | 8.4        | Đọc thêm                      |
| Pl | nần I      | II Khởi động                  |
| 9  | <b>K</b> 1 | ân cận                        |
|    | 9.1        | Giới thiệu                    |
|    | 9.2        | Phân tích toán học            |
|    | 9.3        | Ví dụ trên cơ sở dữ liệu Iris |
|    | 9.4        | Thảo luận                     |
| 10 | Phâ        | ân cụm <i>K</i> -means        |
|    | 10.1       | Giới thiệu 128                |
|    | 10.2       | Phân tích toán học            |
|    | 10.3       | Ví dụ trên Python             |
|    | 10.4       | Phân cụm chữ số viết tay      |
|    | 10.5       | Tách vật thể trong ảnh        |
|    | 10.6       | Nén ảnh                       |
|    | 10.7       | Thảo luận                     |

| 11 | Bộ phân loại naive Bayes<br>145 |                                         |     |
|----|---------------------------------|-----------------------------------------|-----|
|    | 11.1                            | Bộ phân loại naive Bayes                | 145 |
|    | 11.2                            | Các phân phối thường dùng trong NBC<br> | 147 |
|    | 11.3                            | Ví dụ<br>                               | 148 |
|    | 11.4                            | Thảo luận                               | 155 |
|    |                                 | Phần IV Mạng neuron nhân tạo            |     |
| 12 |                                 | Gradient descent<br>158                 |     |
|    | 12.1                            | Giới thiệu<br>                          | 158 |
|    | 12.2                            | Gradient descent cho hàm một biến       | 159 |
|    | 12.3                            | Gradient descent cho hàm nhiều biến     | 164 |
|    | 12.4                            | Gradient descent với momentum           | 167 |
|    | 12.5                            | Nesterov accelerated gradient<br>       | 170 |
|    | 12.6                            | Stochastic gradient descent<br>         | 171 |
|    | 12.7                            | Thảo luận                               | 173 |
| 13 |                                 | Thuật toán học perceptron<br>175        |     |
|    | 13.1                            | Giới thiệu<br>                          | 175 |
|    | 13.2                            | Thuật toán học perceptron<br>           | 176 |
|    | 13.3                            | Ví dụ và minh hoạ trên Python<br>       | 179 |
|    | 13.4                            | Mô hình mạng neuron đầu tiên            | 180 |
|    | 13.5                            | Thảo Luận<br>                           | 183 |

| 14 |      | Hồi quy logistic<br>185                        |     |
|----|------|------------------------------------------------|-----|
|    | 14.1 | Giới thiệu<br>                                 | 185 |
|    | 14.2 | Hàm mất mát và phương pháp tối ưu<br>          | 188 |
|    | 14.3 | Triển khai thuật toán trên Python              | 190 |
|    | 14.4 | Tính chất của hồi quy logistic                 | 193 |
|    | 14.5 | Bài toán phân biệt hai chữ số viết tay<br>     | 195 |
|    | 14.6 | Bài toán phân loại đa lớp                      | 196 |
|    | 14.7 | Thảo luận<br>                                  | 198 |
|    |      |                                                |     |
| 15 |      | Hồi quy softmax<br>201                         |     |
|    | 15.1 | Giới thiệu<br>                                 | 201 |
|    | 15.2 | Hàm softmax<br>                                | 202 |
|    | 15.3 | Hàm mất mát và phương pháp tối ưu<br>          | 205 |
|    | 15.4 | Ví dụ trên Python                              | 211 |
|    | 15.5 | Thảo luận<br>                                  | 213 |
|    |      |                                                |     |
| 16 |      | Mạng neuron đa tầng và lan truyền ngược<br>214 |     |
|    | 16.1 | Giới thiệu<br>                                 | 214 |
|    | 16.2 | Các ký hiệu và khái niệm                       | 217 |
|    | 16.3 | Hàm kích hoạt                                  | 218 |
|    | 16.4 | Lan truyền ngược                               | 220 |
|    | 16.5 | Ví dụ trên Python                              | 225 |
|    | 16.6 | Suy giảm trọng số                              | 230 |
|    | 16.7 | Đọc thêm<br>                                   | 232 |

Phần V Hệ thống gợi ý

| 17 |      | Hệ thống gợi ý dựa trên nội dung<br>234 |     |
|----|------|-----------------------------------------|-----|
|    | 17.1 | Giới thiệu<br>                          | 234 |
|    | 17.2 | Ma trận tiện ích                        | 235 |
|    | 17.3 | Hệ thống dựa trên nội dung              | 237 |
|    | 17.4 | Bài toán MovieLens 100k<br>             | 240 |
|    | 17.5 | Thảo luận                               | 244 |
|    |      |                                         |     |
| 18 |      | Lọc cộng tác lân cận<br>245             |     |
|    | 18.1 | Giới thiệu<br>                          | 245 |
|    | 18.2 | Lọc cộng tác theo người dùng            | 246 |
|    | 18.3 | Lọc cộng tác sản phẩm                   | 251 |
|    | 18.4 | Lập trình trên Python<br>               | 253 |
|    | 18.5 | Thảo luận                               | 256 |
|    |      |                                         |     |
| 19 |      | Lọc cộng tác phân tích ma trận<br>257   |     |
|    | 19.1 | Giới thiệu<br>                          | 257 |
|    | 19.2 | Xây dựng và tối ưu hàm mất mát          | 259 |
|    | 19.3 | Lập trình Python<br>                    | 261 |
|    | 19.4 | Thảo luận                               | 264 |

Phần VI Giảm chiều dữ liệu

| 20 |      | Phân tích giá trị suy biến 266                  |     |
|----|------|-------------------------------------------------|-----|
|    | 20.1 | Giới thiệu<br>                                  | 266 |
|    | 20.2 | Phân tích giá trị suy biến                      | 267 |
|    | 20.3 | Phân tích giá trị suy biến cho bài toán nén ảnh | 271 |
|    | 20.4 | Thảo luận                                       | 273 |
| 21 |      | Phân tích thành phần chính<br>274               |     |
|    | 21.1 | Phân tích thành phần chính                      | 274 |
|    | 21.2 | Các bước thực hiện phân tích thành phần chính   | 279 |
|    | 21.3 | Liên hệ với phân tích giá trị suy biến          | 280 |
|    | 21.4 | Làm thế nào để chọn số chiều của dữ liệu mới    | 282 |
|    | 21.5 | Lưu ý về tính toán phân tích thành phần chính   | 282 |
|    | 21.6 | Một số ứng dụng                                 | 283 |
|    | 21.7 | Thảo luận                                       | 287 |
| 22 |      | Phân tích biệt thức tuyến tính 288              |     |
|    | 22.1 | Giới thiệu<br>                                  | 288 |
|    | 22.2 | Bài toán phân loại nhị phân<br>                 | 289 |
|    | 22.3 | Bài toán phân loại đa lớp                       | 293 |
|    | 22.4 | Ví dụ trên Python                               | 297 |
|    | 22.5 | Thảo luận                                       | 299 |

Phần VII Tối ưu lồi

| 23 |      | Tập lồi và hàm lồi<br>302         |
|----|------|-----------------------------------|
|    | 23.1 | Giới thiệu<br><br>302             |
|    | 23.2 | Tập lồi<br>304                    |
|    | 23.3 | Hàm lồi<br>309                    |
|    | 23.4 | Tóm tắt<br>319                    |
| 24 |      | Bài toán tối ưu lồi<br>320        |
|    | 24.1 | Giới thiệu<br><br>320             |
|    | 24.2 | Nhắc lại bài toán tối ưu<br>324   |
|    | 24.3 | Bài toán tối ưu lồi<br>326        |
|    | 24.4 | Quy hoạch tuyến tính<br>329       |
|    | 24.5 | Quy hoạch toàn phương<br>332      |
|    | 24.6 | Quy hoạch hình học<br><br>334     |
|    | 24.7 | Tóm tắt<br>337                    |
| 25 |      | Đối ngẫu<br>338                   |
|    | 25.1 | Giới thiệu<br><br>338             |
|    | 25.2 | Hàm đối ngẫu Lagrange<br>339      |
|    | 25.3 | Bài toán đối ngẫu Lagrange<br>342 |
|    | 25.4 | Các điều kiện tối ưu<br><br>344   |
|    | 25.5 | Tóm tắt<br>346                    |

Phần VIII Máy vector hỗ trợ

| 26 |      | Máy vector hỗ trợ<br>350                           |     |
|----|------|----------------------------------------------------|-----|
|    | 26.1 | Giới thiệu<br>                                     | 350 |
|    | 26.2 | Xây dựng bài toán tối ưu cho máy vector hỗ trợ     | 352 |
|    | 26.3 | Bài toán đối ngẫu của máy vector hỗ trợ            | 354 |
|    | 26.4 | Lập trình tìm nghiệm cho máy vector hỗ trợ<br>     | 357 |
|    | 26.5 | Tóm tắt                                            | 359 |
| 27 |      | Máy vector hỗ trợ lề mềm                           |     |
|    |      | 361                                                |     |
|    | 27.1 | Giới thiệu<br>                                     | 361 |
|    | 27.2 | Phân tích toán học<br>                             | 362 |
|    | 27.3 | Bài toán đối ngẫu Lagrange<br>                     | 364 |
|    | 27.4 | Bài toán tối ưu không ràng buộc cho SVM lề mềm<br> | 367 |
|    | 27.5 | Lập trình với SVM lề mềm                           | 372 |
|    | 27.6 | Tóm tắt và thảo luận<br>                           | 376 |
| 28 |      | Máy vector hỗ trợ hạt nhân<br>378                  |     |
|    |      |                                                    |     |
|    | 28.1 | Giới thiệu<br>                                     | 378 |
|    | 28.2 | Cơ sở toán học<br>                                 | 380 |
|    | 28.3 | Hàm số hạt nhân                                    | 382 |
|    | 28.4 | Ví dụ minh họa<br>                                 | 384 |
|    | 28.5 | Tóm tắt<br>                                        | 386 |

Mục lục

| 29 | Máy vector hỗ trợ đa lớp<br>387                       |     |
|----|-------------------------------------------------------|-----|
|    | 29.1<br>Giới thiệu<br>                                | 387 |
|    | 29.2<br>Xây dựng hàm mất mát<br>                      | 390 |
|    | 29.3<br>Tính toán giá trị và gradient của hàm mất mát | 393 |
|    | 29.4<br>Thảo luận<br>                                 | 400 |
| A  | Phương pháp nhân tử Lagrange<br>402                   |     |
| B  | Ảnh màu<br>405                                        |     |
|    | Tài liệu tham khảo<br>409                             |     |
|    | Index<br>415                                          |     |

## Chương 0

### <span id="page-14-0"></span>Lời nói đầu

Những năm gần đây, trí tuệ nhân tạo (artificial intelligence, AI) dần nổi lên như một minh chứng cho cuộc cách mạng công nghiệp lần thứ tư, sau động cơ hơi nước, năng lượng điện và công nghệ thông tin. Trí tuệ nhân tạo đã và đang trở thành nhân tố cốt lõi trong các hệ thống công nghệ cao. Thậm chí, nó đã len lỏi vào hầu hết các lĩnh vực của đời sống mà có thể chúng ta không nhận ra. Xe tự hành của Google và Tesla, hệ thống tự tag khuôn mặt trong ảnh của Facebook, trợ lý ảo Siri của Apple, hệ thống gợi ý sản phẩm của Amazon, hệ thống gợi ý phim của Netflix, hệ thống dịch đa ngôn ngữ Google Translate, máy chơi cờ vây AlphaGo và gần đây là AlphaGo Zero của Google DeepMind,... chỉ là một vài ứng dụng nổi bật trong vô vàn những ứng dụng của trí tuệ nhân tạo.

Học máy (machine learning, ML) là một tập con của trí tuệ nhân tạo. Machine learning là một lĩnh vực nhỏ trong khoa học máy tính, có khả năng tự học hỏi dựa trên dữ liệu được đưa vào mà không cần phải được lập trình cụ thể (Machine Learning is the subfield of computer science, that "gives computers the ability to learn without being explicitly programmed" – Wikipedia).

Những năm gần đây, sự phát triển của các hệ thống tính toán cùng lượng dữ liệu khổng lồ được thu thập bởi các hãng công nghệ lớn đã giúp machine learning tiến thêm một bước dài. Một lĩnh vực mới được ra đời được gọi là học sâu (deep learning, DL). Deep learning đã giúp máy tính thực thi những việc vào mười năm trước tưởng chừng là không thể: phân loại cả ngàn vật thể khác nhau trong các bức ảnh, tự tạo chú thích cho ảnh, bắt chước giọng nói và chữ viết, giao tiếp với con người, chuyển đổi ngôn ngữ, hay thậm chí cả sáng tác văn thơ và âm nhạc[1](#page-14-1) .

<span id="page-14-1"></span><sup>1</sup> Đọc thêm: 8 Inspirational Applications of Deep Learning (<https://goo.gl/Ds3rRy>).

<span id="page-15-1"></span>![](_page_15_Figure_1.jpeg)

Hình 0.1. Mối quan hệ giữa AI, ML, và DL. (Nguồn What's the Difference Between Artificial Intelligence, Machine Learning, and Deep Learning? – [https://goo.](https://goo.gl/NNwGCi) [gl/NNwGCi\)](https://goo.gl/NNwGCi).

Mối quan hệ AI-ML-DL

DL là một tập con của ML. ML là một tập con của AI (xem Hình [0.1\)](#page-15-1).

#### <span id="page-15-0"></span>0.1. Mục đích của cuốn sách

Những phát triển thần kỳ của trí tuệ nhân tạo dẫn tới nhu cầu cao về mặt nhần lực làm việc trong các ngành liên quan tới machine learning ở Việt Nam cũng như trên thế giới. Đó cũng là nguồn động lực để tác giả gây dựng và phát triển blog Machine Learning cơ bản từ đầu năm 2017 (<https://machinelearningcoban.com>). Tính tới thời điểm đặt bút viết những dòng này, blog đã có hơn một triệu lượt ghé thăm. Facebook page Machine Learning cơ bản[2](#page-15-2) chạm mốc 14 nghìn lượt likes, Forum Machine Learning cơ bản[3](#page-15-3) đạt tới 17 nghìn thành viên. Trong quá trình viết blog và duy trì các trang Facebook, tác giả đã nhận được nhiều sự ủng hộ của bạn đọc về tinh thần cũng như vật chất. Nhiều bạn đọc cũng khuyến khích tác giả tổng hợp kiến thức trên blog thành một cuốn sách cho cộng đồng những người tiếp cận với ML bằng tiếng Việt. Sự ủng hộ và những lời động viên đó là động lực lớn cho tác giả khi bắt tay vào thực hiện và hoàn thành cuốn sách này.

<span id="page-15-2"></span><sup>2</sup> <https://goo.gl/wyUEjr>

<span id="page-15-3"></span><sup>3</sup> <https://goo.gl/gDPTKX>

Lĩnh vực ML nói chung và DL nói riêng là cực kỳ lớn và có nhiều nhánh nhỏ. Phạm vi một cuốn sách chắc chắn không thể bao quát hết mọi vấn đề và đi sâu vào từng nhánh cụ thể. Do vậy, cuốn sách này chỉ nhằm cung cấp cho bạn đọc những khái niệm, kỹ thuật chung và các thuật toán cơ bản nhất của ML. Từ đó, bạn đọc có thể tự tìm thêm các cuốn sách và khóa học liên quan nếu muốn đi sâu vào từng vấn đề.

Hãy nhớ rằng luôn bắt đầu từ những điều đơn giản. Khi bắt tay vào giải quyết một bài toán ML hay bất cứ bài toán nào, chúng ta nên bắt đầu từ những thuật toán đơn giản. Không phải chỉ có những thuật toán phức tạp mới có thể giải quyết được vấn đề. Những thuật toán phức tạp thường có yêu cầu cao về khả năng tính toán và đôi khi nhạy cảm với cách chọn tham số. Ngược lại, những thuật toán đơn giản giúp chúng ta nhanh chóng có một bộ khung cho mỗi bài toán. Kết quả của các thuật toán đơn giản cũng mang lại cái nhìn sơ bộ về sự phức tạp của mỗi bài toán. Việc cải thiện kết quả sẽ được thực hiện dần ở các bước sau. Cuốn sách này sẽ trang bị cho bạn đọc những kiến thức khái quát và một số hướng tiếp cận cơ bản cho các bài toán ML. Để tạo ra các sản phẩm thực tiễn, chúng ta cần học hỏi và thực hành thêm nhiều.

#### <span id="page-16-0"></span>0.2. Hướng tiếp cận của cuốn sách

Để giải quyết mỗi bài toán ML, chúng ta cần chọn một mô hình phù hợp. Mô hình này được mô tả bởi bộ các tham số mà chúng ta cần đi tìm. Thông thường, lượng tham số có thể lên tới hàng triệu và được tìm bằng cách giải một bài toán tối ưu.

Khi viết về các thuật toán ML, tác giả sẽ bắt đầu từ những ý tưởng trực quan. Những ý tưởng này được mô hình hoá dưới dạng một bài toán tối ưu. Các suy luận toán học và ví dụ mẫu trên Python ở cuối mỗi chương sẽ giúp bạn đọc hiểu rõ hơn về nguồn gốc, ý nghĩa, và cách sử dụng mỗi thuật toán. Xen kẽ giữa những thuật toán ML, tác giả sẽ trình bày các kỹ thuật tối ưu cơ bản, với hy vọng giúp bạn đọc hiểu rõ hơn bản chất của vấn đề.

#### <span id="page-16-1"></span>0.3. Đối tượng của cuốn sách

Cuốn sách được thực hiện hướng tới nhiều nhóm độc giả khác nhau. Nếu bạn không thực sự muốn đi sâu vào phần toán, bạn vẫn có thể tham khảo mã nguồn và cách sử dụng các thư viện. Nhưng để sử dụng các thư viện một cách hiệu quả, bạn cũng cần hiểu nguồn gốc của mô hình và ý nghĩa của các tham số. Còn nếu bạn thực sự muốn tìm hiểu nguồn gốc, ý nghĩa của các thuật toán, bạn có thể học được nhiều điều từ cách xây dựng và tối ưu các mô hình. Phần tổng hợp các kiến thức toán cần thiết trong Phần I sẽ là một nguồn tham khảo súc tích bất cứ khi nào bạn có thắc mắc về các dẫn giải toán học. Phần VII được dành riêng để nói về tối ưu lồi – một mảng quan trọng trong tối ưu, phù hợp với các bạn thực sự muốn đi sâu thêm về tối ưu.

Các dẫn giải toán học được xây dựng phù hợp với chương trình toán phổ thông và đại học ở Việt Nam. Các từ khóa khi được dịch sang tiếng Việt đều dựa trên những tài liệu tác giả được học trong nhiều năm tại Việt Nam.

Phần cuối cùng của sách có mục Index các thuật ngữ quan trọng và thuật ngữ tiếng Anh đi kèm giúp bạn dần làm quen khi đọc các tài liệu tiếng Anh.

#### <span id="page-17-0"></span>0.4. Yêu cầu về kiến thức

Để có thể bắt đầu đọc cuốn sách này, bạn cần có một kiến thức nhất định về đại số tuyến tính, giải tích ma trận, xác suất thống kê, và kỹ năng lập trình.

Phần I của cuốn sách ôn tập lại các kiến thức toán quan trọng được dùng trong ML. Khi gặp khó khăn về toán, bạn được khuyến khích đọc lại các chương trong phần này.

Ngôn ngữ lập trình được sử dụng trong cuốn sách là Python. Python là một ngôn ngữ lập trình miễn phí, có thể được cài đặt dễ dàng trên các nền tảng hệ điều hành khác nhau. Quan trọng hơn, có rất nhiều thư viện hỗ trợ ML cũng như DL trên Python. Có hai thư viện Python chính thường được sử dụng trong cuốn sách là numpy và scikit-learn.

Numpy (<http://www.numpy.org/>) là một thư viện phổ biến giúp xử lý các phép toán liên quan đến các mảng nhiều chiều, hỗ trợ các hàm gần gũi với đại số tuyến tính. Nếu bạn đọc chưa quen thuộc với numpy, bạn có thể tham gia một khóa học ngắn miễn phí trên trang web kèm theo cuốn sách này (<https://fundaml.com>). Bạn sẽ được làm quen với cách xử lý các mảng nhiều chiều với nhiều ví dụ và bài tập thực hành. Các kỹ thuật xử lý mảng trong cuốn sách này đều được đề cập tại đây.

Scikit-learn, hay sklearn (<http://scikit-learn.org/>), là một thư viện chứa đầy đủ các thuật toán ML cơ bản và rất dễ sử dụng. Tài liệu của scikit-learn cũng là một nguồn tham khảo chất lượng cho các bạn làm ML. Scikit-learn sẽ được dùng trong cuốn sách để kiểm chứng các suy luận toán học và các mô hình được xây dựng thông qua numpy.

<span id="page-17-1"></span>Có rất nhiều thư viện giúp chúng ta tạo ra các sản phẩm ML/DL mà không yêu cầu nhiều kiến thức toán. Tuy nhiên, cuốn sách này hướng tới việc giúp bạn đọc hiểu bản chất toán học đằng sau mỗi mô hình trước khi áp dụng các thư viện sẵn có. Việc sử dụng thư viện cũng yêu cầu kiến thức nhất định về việc lựa chọn mô hình và điều chỉnh các tham số.

## 0.5. Mã nguồn đi kèm

Toàn bộ mã nguồn trong cuốn sách có thể được tìm thấy tại [https://goo.gl/](https://goo.gl/Fb2p4H) [Fb2p4H](https://goo.gl/Fb2p4H). Các file có đuôi .ipynb là các Jupyter notebook chứa mã nguồn. Các file có đuôi .pdf, và .png là các hình vẽ được sử dụng trong cuốn sách.

### <span id="page-18-0"></span>0.6. Bố cục của cuốn sách

Cuốn sách này được chia thành 8 phần và sẽ tiếp tục được cập nhật:

Phần I ôn tập lại những kiến thức quan trọng trong đại số tuyến tính, giải tích ma trận, xác suất, và hai phương pháp phổ biến trong việc ước lượng tham số cho các mô hình ML dựa trên thống kê.

Phần II giới thiệu các khái niệm cơ bản trong ML, các kỹ thuật xây dựng vector đặc trưng cho dữ liệu, một mô hình ML cơ bản – hồi quy, và một hiện tượng cần tránh khi xây dựng các mô hình ML.

Phần III giúp các bạn làm quen với các mô hình ML không yêu cầu nhiều kiến thức toán phức tạp. Qua đây, bạn đọc sẽ có cái nhìn sơ bộ về việc xây dựng các mô hình ML.

Phần IV đề cập tới một nhóm các thuật toán ML phổ biến nhất – mạng neuron nhân tạo, là nền tảng cho các mô hình DL phức tạp hiện nay. Phần này cũng giới thiệu một kỹ thuật tối ưu phổ biến cho các bài toán tối ưu không ràng buộc.

Phần V giới thiệu về các kỹ thuật thường dùng trong các hệ thống gợi ý sản phẩm.

Phần VI giới thiệu các kỹ thuật giảm chiều dữ liệu.

Phần VII trình bày cụ thể hơn về tối ưu, đặc biệt là tối ưu lồi. Các bài toán tối ưu lồi có ràng buộc cũng được giới thiệu trong phần này.

<span id="page-18-1"></span>Phần VIII giới thiệu các thuật toán phân loại dựa trên máy vector hỗ trợ.

## 0.7. Các lưu ý về ký hiệu

Các ký hiệu toán học trong sách được mô tả ở Bảng [0.1](#page-21-0) và đầu Chương 1. Các khung với font chữ có cùng chiều rộng được dùng để chứa các đoạn mã nguồn.

text in a box with constant width represents source codes.

Các đoạn ký tự với constant width (có cùng chiều rộng) được dùng để chỉ các biến, hàm số, chuỗi,... trong các đoạn mã.

Đóng khung và in nghiêng

Các khái niệm, định nghĩa, định lý, và lưu ý quan trọng được đóng khung và in nghiêng.

Ký tự phân cách giữa phần nguyên và phần thập phân của các số thực là dấu chấm (.) thay vì dấu phẩy (,) như trong các tài liệu tiếng Việt khác. Cách làm này thống nhất với các tài liệu tiếng Anh và các ngôn ngữ lập trình.

### <span id="page-19-0"></span>0.8. Tham khảo thêm

Có nhiều cuốn sách, khóa học, website hay về machine learning/deep learning. Trong đó, tôi xin đặc biệt nhấn mạnh các nguồn tham khảo sau:

### 0.8.1. Khoá học

- a. Khoá học Machine Learning của Andrew Ng trên Coursera ([https://goo.gl/](https://goo.gl/WBwU3K) [WBwU3K](https://goo.gl/WBwU3K)).
- b. Khoá học mới Deep Learning Specialization cũng của Andrew Ng trên Coursera (<https://goo.gl/ssXfYN>).
- c. Các khóa CS224n: Natural Language Processing with Deep Learning ([https:](https://goo.gl/6XTNkH) [//goo.gl/6XTNkH](https://goo.gl/6XTNkH)); CS231n: Convolutional Neural Networks for Visual Recognition (<http://cs231n.stanford.edu/>); CS246: Mining Massive Data Sets ([https:](https://goo.gl/TEMQ9H) [//goo.gl/TEMQ9H](https://goo.gl/TEMQ9H)) của Stanford.

### 0.8.2. Sách

- a. C. Bishop, Pattern Recognition and Machine Learning (<https://goo.gl/pjgqRr>), Springer, 2006 [Bis06].
- b. I. Goodfellow et al., Deep Learning (<https://goo.gl/sXaGwV>), MIT press, 2016 [GBC16].
- c. J. Friedman et al., The Elements of Statistical Learning ([https://goo.gl/](https://goo.gl/Qh9EkB) [Qh9EkB](https://goo.gl/Qh9EkB)), Springer, 2001 [FHT01].
- d. Y. Abu-Mostafa et al., Learning from data (<https://goo.gl/SRfNFJ>), AML-Book New York, 2012 [AMMIL12].
- e. S. JD Prince, Computer Vision: Models, Learning, and Inference ([https://goo.](https://goo.gl/9Fchf3) [gl/9Fchf3](https://goo.gl/9Fchf3)), Cambridge University Press, 2012 [Pri12].

f. S. Boyd et al., Convex Optimization (<https://goo.gl/NomDpC>), Cambridge university press, 2004 [BV04].

Ngoài ra, các website Machine Learning Mastery (<https://goo.gl/5DwGbU>), Pyimagesearch (<https://goo.gl/5DwGbU>). Kaggle (<https://www.kaggle.com/>), Scikitlearn (<http://scikit-learn.org/>) cũng là các nguồn thông tin hữu ích.

#### <span id="page-20-0"></span>0.9. Đóng góp ý kiến

Các bạn có thể gửi các đóng góp tới địa chỉ email vuhuutiep@gmail.com hoặc tạo một GitHub issue mới tại <https://goo.gl/zPYWKV>.

#### <span id="page-20-1"></span>0.10. Lời cảm ơn

Trước hết, tôi xin được cảm ơn sự ủng hộ và chia sẻ nhiệt tình của bạn bè trên Facebook từ những ngày đầu ra mắt blog. Xin được gửi lời cảm ơn chân thành tới bạn đọc Machine Learning cơ bản đã đồng hành trong hơn một năm qua.

Tôi cũng may mắn nhận được những góp ý và phản hồi tích cực từ các thầy cô tại các trường đại học lớn trong và ngoài nước. Xin phép được gửi lời cảm ơn tới thầy Phạm Ngọc Nam và cô Nguyễn Việt Hương (ĐH Bách Khoa Hà Nội), thầy Chế Viết Nhật Anh (ĐH Bách Khoa TP.HCM), thầy Nguyễn Thanh Tùng (ĐH Thuỷ Lợi), và thầy Trần Duy Trác (ĐH Johns Hopkins).

Đặc biệt, xin cảm ơn Nguyễn Hoàng Linh và Hoàng Đức Huy, Đại học Waterloo, Canada đã nhiệt tình giúp tôi xây dựng trang <FundaML.com>, cho phép độc giả học Python/Numpy trực tiếp trên trình duyệt. Xin cảm ơn các bạn Nguyễn Tiến Cường, Nguyễn Văn Giang, Vũ Đình Quyền, Lê Việt Hải, và Đinh Hoàng Phong đã góp ý sửa đổi nhiều điểm trong các bản nháp.

Ngoài ta, cũng xin cảm ơn những người bạn thân của tôi tại Penn State (ĐH bang Pennsylvania) đã luôn bên cạnh tôi trong thời gian tôi thực hiện dự án, bao gồm gia đình anh Triệu Thanh Quang, gia đình anh Trần Quốc Long, bạn thân Nguyễn Phương Chi, và các đồng nghiệp tại Phòng nghiên cứu Xử lý Thông tin và Thuật toán (Information Processing and Algorithm Laboratory, iPAL).

Cuối cùng và quan trọng nhất, xin gửi lời cảm ơn sâu sắc nhất tới gia đình tôi, những người luôn ủng hộ vô điều kiện và hỗ trợ tôi hết mình trong quá trình thực hiện dự án này.

#### <span id="page-20-2"></span>0.11. Bảng các ký hiệu

Các ký hiệu sử dụng trong sách được liệt kê trong Bảng [0.1](#page-21-0) ở trang tiếp theo.

Bảng 0.1: Các quy ước ký hiệu và tên gọi được sử dụng trong sách

<span id="page-21-0"></span>

|                                                       | [z .                                                                             |
|-------------------------------------------------------|----------------------------------------------------------------------------------|
| Ký hiệu                                               | Ý nghĩa                                                                          |
| x, y, N, k                                            | in nghiêng, thường hoặc hoa, là các số vô hướng                                  |
| $\mathbf{x}, \mathbf{y}$                              | in đậm, chữ thường, là các vector                                                |
| $\mathbf{X}, \mathbf{Y}$                              | in đậm, chữ hoa, là các ma trận                                                  |
| $\mathbb{R}$                                          | tập hợp các số thực                                                              |
| N                                                     | tập hợp các số tự nhiên                                                          |
| $\mathbb{C}$                                          | tập hợp các số phức                                                              |
| $\mathbb{R}^m$                                        | tập hợp các vector thực có $m$ phần tử                                           |
| $\mathbb{R}^{m \times n}$                             | tập hợp các ma trận thực có $m$ hàng, $n$ cột                                    |
| $\mathbb{S}^n$                                        | tập hợp các ma trận vuông đối xứng bậc $n$                                       |
| $\mathbb{S}^n_+$                                      | tập hợp các ma trận nửa xác định dương bậc $n$                                   |
| $\mathbb{S}^n_{++}$                                   | tập hợp các ma trận xác định dương bậc $n$                                       |
| $\in$                                                 | phần tử thuộc tập hợp                                                            |
| ∃                                                     | tồn tại                                                                          |
| A                                                     | mọi                                                                              |
| ≜                                                     | ký hiệu là/bởi. Ví dụ $a \triangleq f(x)$ nghĩa là "ký hiệu $f(x)$ bởi $a$ ".    |
| $x_i$                                                 | phần tử thứ $i$ (tính từ 1) của vector $\mathbf{x}$                              |
| sgn(x)                                                | hàm xác định dấu. Bằng 1 nếu $x \ge 0$ , bằng -1 nếu $x < 0$ .                   |
| $\exp(x)$                                             | $e^x$                                                                            |
| $\log(x)$                                             | logarit $t\psi$ $nhi\hat{e}n$ của số thực dương $x$                              |
| $\underset{x}{\operatorname{argmin}} f(x)$            | giá trị của $x$ để hàm $f(x)$ đạt giá trị nhỏ nhất                               |
| $\operatorname*{argmax}_{x} f(x)$                     | giá trị của $x$ để hàm $f(x)$ đạt giá trị lớn nhất                               |
| $\begin{aligned} a_{ij} \ \mathbf{A}^T \end{aligned}$ | phần tử hàng thứ $i$ , cột thứ $j$ của ma trận ${\bf A}$                         |
|                                                       | chuyển vị của ma trận ${\bf A}$                                                  |
| $\mathbf{A}^{H}$                                      | chuyển vị liên hợp (Hermitian) của ma trận phức <b>A</b>                         |
| $\mathbf{A}^{-1}$                                     | nghịch đảo của ma trận vuông ${\bf A},$ nếu tồn tại                              |
| ${\bf A}^\dagger$                                     | giả nghịch đảo của ma trận không nhất thiết vuông <b>A</b>                       |
| $\mathbf{A}^{-T}$                                     | chuyển vị của nghịch đảo của ma trận ${\bf A}$ , nếu tồn tại                     |
| $\ \mathbf{x}\ _p$                                    | $\ell_p$ norm của vector ${\bf x}$                                               |
| $\ \mathbf{A}\ _F$                                    | Frobenius norm của ma trận <b>A</b>                                              |
| $\operatorname{diag}(\mathbf{A})$                     | đường chéo chính của ma trận ${\bf A}$                                           |
| $\mathrm{trace}(\mathbf{A})$                          | trace của ma trận <b>A</b>                                                       |
| $\det(\mathbf{A})$                                    | định thức của ma trận vuông ${\bf A}$                                            |
| $rank(\mathbf{A})$                                    | hạng của ma trận ${\bf A}$                                                       |
| o.w                                                   | otherwise – trong các trường hợp còn lại                                         |
| $\frac{\partial f}{\partial x}$                       | đạo hàm của hàm số $f$ theo $x \in \mathbb{R}$                                   |
| $\nabla_{\mathbf{x}} f$                               | gradient của hàm số $f$ theo $\mathbf{x}$ ( $\mathbf{x}$ là vector hoặc ma trận) |
| $\nabla^2_{\mathbf{x}} f$                             | gradient bậc hai của hàm số $f$ theo $\mathbf{x}$ , còn được gọi là $Hesse$      |
| •                                                     | Hadamard product (elemenwise product). Phép nhân từng phần tử                    |
|                                                       | của hai vector hoặc ma trận cùng kích thước.                                     |
| $\propto$                                             | tỉ lệ với                                                                        |
|                                                       | đường nét liền                                                                   |
|                                                       | đường nét đứt                                                                    |
|                                                       | đường nét chấm (đường chấm chấm)                                                 |
|                                                       | đường chấm gạch<br>nền chấm                                                      |
|                                                       |                                                                                  |
|                                                       | nền sọc chéo                                                                     |