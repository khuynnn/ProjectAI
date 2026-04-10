# Phần VIII

Máy vector hỗ trợ

Máy vector hỗ trợ

## 26.1. Giới thiệu

 $M\acute{a}y~vector~h\~o~trợ~(support~vector~machine, SVM)$  là một trong những thuật toán phân loại phổ biến và hiệu quả. Ý tưởng đứng sau SVM khá đơn giản, nhưng để giải bài toán tối ưu SVM, chúng ta cần kiến thức về tối ưu và đối ngẫu.

Trước khi đi vào phần ý tưởng chính của SVM, chúng ta cùng ôn lại kiến thức về hình học giải tích trong chương trình phổ thông.

### 26.1.1. Khoảng cách từ một điểm tới một siêu mặt phẳng

Trong không gian hai chiều, khoảng cách từ một điểm có toạ độ  $(x_0, y_0)$  tới đường thẳng có phương trình  $w_1x + w_2y + b = 0$  được xác định bởi

$$\frac{|w_1x_0 + w_2y_0 + b|}{\sqrt{w_1^2 + w_2^2}}$$

Trong không gian ba chiều, khoảng cách từ một điểm có toạ độ  $(x_0, y_0, z_0)$  tới một mặt phẳng có phương trình  $w_1x + w_2y + w_3z + b = 0$  được xác định bởi

$$\frac{|w_1x_0 + w_2y_0 + w_3z_0 + b|}{\sqrt{w_1^2 + w_2^2 + w_3^2}}$$

Hơn nữa, nếu bỏ dấu trị tuyệt đối ở tử số, ta có thể xác định được điểm đó nằm về phía nào của đường thẳng hay mặt phẳng đang xét. Những điểm làm cho biểu thức trong dấu trị tuyệt đối mang dấu dương nằm về cùng một phía (tạm gọi là  $phía\ dương$ ), những điểm làm cho giá trị này mang dấu âm nằm về phía còn lại (gọi là  $phía\ \hat{a}m$ ). Những điểm làm cho tử số bằng không sẽ nằm trên đường thẳng/mặt phẳng phân chia.

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

**Hình 26.1.** Hai lớp dữ liệu vuông và tròn là tách biệt tuyến tính. Có vô số đường thẳng có thể phân loại chính xác hai lớp dữ liệu này (xem thêm Chương 13).

Các công thức này có thể được tổng quát lên cho trường hợp không gian d chiều. Khoảng cách từ một điểm (vector) có toạ độ  $(x_{10},x_{20},\ldots,x_{d0})$  tới siêu phẳng  $w_1x_1+w_2x_2+\cdots+w_dx_d+b=0$  được xác định bởi

$$\frac{|w_1 x_{10} + w_2 x_{20} + \dots + w_d x_{d0} + b|}{\sqrt{w_1^2 + w_2^2 + \dots + w_d^2}} = \frac{|\mathbf{w}^T \mathbf{x}_0 + b|}{\|\mathbf{w}\|_2}$$

với 
$$\mathbf{x}_0 = [x_{10}, x_{20}, \dots, x_{d0}]^T, \mathbf{w} = [w_1, w_2, \dots, w_d]^T.$$

### 26.1.2. Nhắc lại bài toán phân loại hai lớp dữ liêu

Xin nhắc lại bài toán phân loại đã đề cập trong Chương 13 (PLA). Giả sử có hai lớp dữ liệu được mô tả bởi các vector đặc trưng trong không gian nhiều chiều. Hơn nữa, hai lớp dữ liệu này là tách biệt tuyến tính, tức tồn tại một siêu phẳng phân chia chính xác hai lớp đó. Hãy tìm một siêu phẳng sao cho tất cả các điểm thuộc một lớp nằm về cùng một phía của siêu phẳng đó và ngược phía với toàn bộ các điểm thuộc lớp còn lại. Chúng ta đã biết rằng, thuật toán PLA có thể thực hiện được việc này nhưng PLA có thể cho vô số nghiệm như Hình 26.1.

Có một câu hỏi được đặt ra: Trong vô số các mặt phân chia đó, đâu là mặt tốt nhất? Trong ba đường thẳng minh họa trong Hình 26.1, có hai đường thẳng khá lệch về phía lớp hình tròn. Điều này có thể khiến nhiều điểm hình tròn chưa nhìn thấy bị phân loại lỗi thành điểm hình vuông. Liệu có cách nào tìm được đường phân chia sao cho đường này không lệch về một lớp không?

Để trả lời câu hỏi này, chúng ta cần tìm một tiêu chuẩn để đo sự  $l\hat{e}ch$  về mỗi lớp của đường phân chia. Gọi khoảng cách nhỏ nhất từ một điểm thuộc một lớp tới đường phân chia là  $l\hat{e}$  (margin). Ta cần tìm một đường phân chia sao cho lề của hai lớp là như nhau đối với đường phân chia đó. Hơn nữa, độ rộng của lề càng lớn thì khả năng xảy ra phân loại lỗi càng thấp. Bài toán tối ưu trong SVM chính là bài toán đi tìm đường phân chia sao cho lề rộng nhất. Đây cũng là lý do SVM còn được gọi là  $b\hat{\rho}$  phân loại  $l\hat{e}$  lớn nhất (maximum margin classifier). Nguồn gốc tên gọi máy vector  $h\tilde{o}$   $tr\phi$  sẽ sớm được làm sáng tỏ.

![](_page_3_Figure_1.jpeg)

![](_page_3_Figure_2.jpeg)

Hình 26.2. Ý tưởng của SVM. Lề của một lớp được định nghĩa là khoảng cách từ các điểm gần nhất của lớp đó tới mặt phân chia. Lề của hai lớp phải bằng nhau và lớn nhất có thể.

<span id="page-3-0"></span>![](_page_3_Figure_4.jpeg)

Hình 26.3. Giả sử mặt phân chia có phương trình w<sup>T</sup> x + b = 0. Không mất tính tổng quát, bằng cách nhân các hệ số w và b với các hằng số phù hợp, ta có thể giả sử rằng điểm gần nhất của lớp vuông tới mặt này thoả mãn w<sup>T</sup> x+b = 1. Khi đó, điểm gần nhất của lớp tròn thoả mãn w<sup>T</sup> x + b = −1.

## 26.2. Xây dựng bài toán tối ưu cho máy vector hỗ trợ

Giả sử dữ liệu trong tập huấn luyện là các cặp (vector đặc trưng, nhãn): (x1, y1),(x2, y2), . . . ,(x<sup>N</sup> , y<sup>N</sup> ) nhãn bằng +1 hoặc -1 và N là số điểm dữ liệu. Không mất tính tổng quát, giả sử các điểm vuông có nhãn là 1, các điểm tròn có nhãn là -1 và siêu phẳng w<sup>T</sup> x + b = 0 là mặt phân chia hai lớp (Hình [26.3\)](#page-3-0). Ngoài ra, lớp hình vuông nằm về phía dương, lớp hình tròn nằm về phía âm của mặt phân chia. Nếu xảy ra điều ngược lại, ta chỉ cần đổi dấu của w và b. Bài toán tối ưu trong SVM sẽ là bài toán đi tìm các tham số mô hình w và b.

Với cặp dữ liệu (xn, yn) bất kỳ, khoảng cách từ x<sup>n</sup> tới mặt phân chia là <sup>y</sup>n(w<sup>T</sup> <sup>x</sup>n+b) kwk<sup>2</sup> . Điều này xảy ra ta đã giả sử y<sup>n</sup> cùng dấu với phía của xn. Từ đó suy ra y<sup>n</sup> cùng dấu với (w<sup>T</sup> x<sup>n</sup> + b) và tử số luôn là một đại lượng không âm. Với mặt phân chia này, lề được tính là khoảng cách gần nhất từ một điểm (trong cả hai lớp, vì cuối cùng lề của hai lớp bằng nhau) tới mặt phân chia:

$$\hat{\hat{\mathbf{e}}} = \min_{n} \frac{y_n(\mathbf{w}^T \mathbf{x}_n + b)}{\|\mathbf{w}\|_2}$$

Bài toán tối ưu của SVM đi tìm  ${\bf w}$  và b sao cho lề đạt giá trị lớn nhất:

<span id="page-4-0"></span>
$$(\mathbf{w}, b) = \arg\max_{\mathbf{w}, b} \left\{ \min_{n} \frac{y_n(\mathbf{w}^T \mathbf{x}_n + b)}{\|\mathbf{w}\|_2} \right\} = \arg\max_{\mathbf{w}, b} \left\{ \frac{1}{\|\mathbf{w}\|_2} \min_{n} y_n(\mathbf{w}^T \mathbf{x}_n + b) \right\}$$
(26.1)

Nếu ta thay vector trọng số  $\mathbf{w}$  bởi  $k\mathbf{w}$  và b bởi kb trong đó k là một hằng số dương bất kỳ thì mặt phân chia không thay đổi, tức khoảng cách từ từng điểm đến mặt phân chia không đổi, tức lề không đổi. Vì vậy, ta có thể giả sử:

$$y_m(\mathbf{w}^T\mathbf{x}_m + b) = 1$$

với những điểm nằm gần mặt phân chia nhất (được khoanh tròn trong Hình 26.3).

Như vậy, với mọi n ta luôn có

$$y_n(\mathbf{w}^T\mathbf{x}_n + b) \ge 1$$

Bài toán tối ưu (26.1) có thể được đưa về bài toán tối ưu ràng buộc có dạng

$$(\mathbf{w}, b) = \arg \max_{\mathbf{w}, b} \frac{1}{\|\mathbf{w}\|_2}$$
thoả mãn:  $y_n(\mathbf{w}^T \mathbf{x}_n + b) \ge 1, \forall n = 1, 2, \dots, N$  (26.2)

Bằng một biến đổi đơn giản, ta có thể tiếp tục đưa bài toán này về dạng

<span id="page-4-1"></span>
$$(\mathbf{w}, b) = \arg\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_{2}^{2}$$
thoả mãn:  $1 - y_{n}(\mathbf{w}^{T}\mathbf{x}_{n} + b) \leq 0, \forall n = 1, 2, \dots, N$  (26.3)

 $\mathring{O}$  đây, ta đã lấy nghịch đảo hàm mục tiêu, bình phương nó để được một hàm khả vi, và nhân với  $\frac{1}{2}$  để biểu thức đạo hàm đẹp hơn.

Trong bài toán (26.3), hàm mục tiêu là một chuẩn – có dạng toàn phương. Các hàm bất phương trình ràng buộc là affine. Vậy bài toán (26.3) là một bài toán quy hoạch toàn phương. Hơn nữa, hàm mục tiêu là lồi chặt vì  $\|\mathbf{w}\|_2^2 = \mathbf{w}^T \mathbf{I} \mathbf{w}$  và  $\mathbf{I}$  là ma trận đơn vị – một ma trận xác định dương. Từ đây có thể suy ra nghiệm của SVM là duy nhất.

Tới đây, bài toán này có thể giải được bằng các công cụ hỗ trợ giải quy hoạch toàn phương, ví dụ CVXOPT. Tuy nhiên, việc giải bài toán này trở nên phức tạp khi số chiều d của không gian dữ liệu và số điểm dữ liệu N lớn. Thay vào đó, người ta thường giải bài toán đối ngẫu của bài toán này. Thứ nhất, bài toán đối ngẫu có những tính chất thú vị khiến nó được giải một cách hiệu quả hơn. Thứ hai, trong quá trình xây dựng bài toán đối ngẫu, người ta thấy rằng SVM có thể được áp dụng cho những bài toán mà dữ liệu không nhất thiết tách biệt tuyến tính, như chúng ta sẽ thấy ở các chương sau của phần này.

Xác đinh lớp cho một điểm dữ liệu mới

Sau khi đã tìm được mặt phân chia  $\mathbf{w}^T\mathbf{x}+b=0$ , nhãn của một điểm bất kỳ sẽ được xác định đơn giản bằng

$$class(\mathbf{x}) = sgn(\mathbf{w}^T \mathbf{x} + b)$$

## 26.3. Bài toán đối ngẫu của máy vector hỗ trợ

Bài toán tối ưu (26.3) là một bài toán lồi. Chúng ta biết rằng nếu một bài toán lồi thoả mãn tiêu chuẩn Slater thì đối ngẫu mạnh xảy ra (xem Mục 25.3.2). Ngoài ra, nếu đối ngẫu mạnh thoả mãn thì nghiệm của bài toán chính là nghiệm của hệ điều kiện KKT (xem Mục 25.4.2).

### 26.3.1. Kiểm tra tiêu chuẩn Slater

Trong bước này, chúng ta sẽ chứng minh bài toán tối ưu (26.3) thoả mãn điều kiện Slater. Điều kiện Slater nói rằng, nếu tồn tại  $\mathbf{w}, b$  thoả mãn

$$1 - y_n(\mathbf{w}^T \mathbf{x}_n + b) < 0, \quad \forall n = 1, 2, \dots, N$$

thì đối ngẫu mạnh cũng thoả mãn. Việc kiểm tra điều kiện này không quá phức tạp. Vì luôn có một siêu phẳng phân chia hai lớp dữ liệu tách biệt tuyến tính nên tập khả thi của bài toán tối ưu (26.3) khác rỗng. Điều này cũng có nghĩa là luôn tồn tại cặp  $(\mathbf{w}_0,b_0)$  sao cho:

$$1 - y_n(\mathbf{w}_0^T \mathbf{x}_n + b_0) \le 0, \quad \forall n = 1, 2, \dots, N$$
 (26.4)

$$\Leftrightarrow 2 - y_n(2\mathbf{w}_0^T\mathbf{x}_n + 2b_0) \le 0, \quad \forall n = 1, 2, \dots, N$$
(26.5)

Vậy chỉ cần chọn  $\mathbf{w}_1=2\mathbf{w}_0$  và  $b_1=2b_0,$  ta sẽ có:

$$1 - y_n(\mathbf{w}_1^T \mathbf{x}_n + b_1) \le -1 < 0, \quad \forall n = 1, 2, \dots, N$$

Điều này chỉ ra rằng  $(\mathbf{w}_1, b_1)$  là một điểm khả thi chặt. Từ đó suy ra điều kiện Slater thoả mãn.

### 26.3.2. Hàm Lagrange của bài toán tối ưu

Hàm Lagrange của bài toán (26.3) là

<span id="page-5-0"></span>
$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\lambda}) = \frac{1}{2} \|\mathbf{w}\|_{2}^{2} + \sum_{n=1}^{N} \lambda_{n} (1 - y_{n}(\mathbf{w}^{T} \mathbf{x}_{n} + b))$$
 (26.6)

với 
$$\boldsymbol{\lambda} = [\lambda_1, \lambda_2, \dots, \lambda_N]^T$$
 và  $\lambda_n \ge 0, \ \forall n = 1, 2, \dots, N.$ 

### 26.3.3. Hàm đối ngẫu Lagrange

Theo định nghĩa, hàm đối ngẫu Lagrange là

$$g(\lambda) = \min_{\mathbf{w}, b} \mathcal{L}(\mathbf{w}, b, \lambda)$$

với λ 0. Việc tìm giá trị nhỏ nhất của hàm này theo w và b có thể đựợc thực hiện bằng cách giải hệ phương trình đạo hàm của L(w, b,λ) theo w và b bằng 0:

<span id="page-6-0"></span>
$$\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}, b, \lambda) = \mathbf{w} - \sum_{n=1}^{N} \lambda_n y_n \mathbf{x}_n = \mathbf{0} \Rightarrow \mathbf{w} = \sum_{n=1}^{N} \lambda_n y_n \mathbf{x}_n$$
 (26.7)

$$\nabla_b \mathcal{L}(\mathbf{w}, b, \boldsymbol{\lambda}) = \sum_{n=1}^N \lambda_n y_n = 0$$
 (26.8)

Thay [\(26.7\)](#page-6-0) và [\(26.8\)](#page-6-0) vào [\(26.6\)](#page-5-0) ta thu được g(λ) [63](#page-6-1):

<span id="page-6-2"></span>
$$g(\lambda) = \sum_{n=1}^{N} \lambda_n - \frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} \lambda_n \lambda_m y_n y_m \mathbf{x}_n^T \mathbf{x}_m$$
 (26.9)

Hàm g(λ) trong [\(26.9\)](#page-6-2) là hàm số quan trọng nhất của SVM, chúng ta sẽ thấy rõ hơn ở Chương 28.

Ta có thể viết lại g(λ) dưới dạng[64](#page-6-3)

$$g(\lambda) = -\frac{1}{2} \lambda^T \mathbf{V}^T \mathbf{V} \lambda + \mathbf{1}^T \lambda.$$
 (26.10)

với 
$$\mathbf{V} = \begin{bmatrix} y_1 \mathbf{x}_1, y_2 \mathbf{x}_2, \dots, y_N \mathbf{x}_N \end{bmatrix}$$
 và  $\mathbf{1} = [1, 1, \dots, 1]^T$ .

Nếu đặt K = V<sup>T</sup>V thì K là một ma trận nửa xác định dương. Thật vậy, với mọi vector λ ta có λ <sup>T</sup>Kλ = λ <sup>T</sup>V<sup>T</sup>Vλ = kVλk 2 <sup>2</sup> ≥ 0. Vậy g(λ) = − 1 2 λ <sup>T</sup>Kλ + 1 <sup>T</sup>λ là một hàm lõm.

### 26.3.4. Bài toán đối ngẫu Lagrange

Từ đó, kết hợp hàm đối ngẫu Lagrange và các điều kiện ràng buộc của λ, ta sẽ thu được bài toán đối ngẫu Lagrange của bài toán [\(26.3\)](#page-4-1):

<span id="page-6-4"></span>
$$\lambda = \arg\max_{\lambda} g(\lambda)$$
 thoả mãn:  $\lambda \succeq 0$  
$$\sum_{n=1}^{N} \lambda_n y_n = 0$$
 (26.11)

<span id="page-6-1"></span><sup>63</sup> Phần chứng minh coi như một bài tập nhỏ cho bạn đọc.

<span id="page-6-3"></span><sup>64</sup> Phần chứng minh coi như một bài tập nhỏ khác cho bạn đọc.

Ràng buộc thứ hai được lấy từ [\(26.8\)](#page-6-0). Đây là một bài toán lồi vì ta đang đi tìm giá trị lớn nhất của một hàm mục tiêu lõm trên một đa diện. Hơn nữa, đây là một bài toán quy hoạch toàn phương và cũng có thể được giải bằng các thư viện như CVXOPT.

Biến tối ưu trong bài toán tối ngẫu là λ, là một vector N chiều tương ứng với số điểm dữ liệu. Trong khi đó, số tham số phải tìm trong bài toán tối ưu chính [\(26.3\)](#page-4-1) là d + 1, chính là tổng số chiều của w và b, tức số chiều của mỗi điểm dữ liệu cộng một. Trong rất nhiều trường hợp, số điểm dữ liệu trong tập huấn luyện lớn hơn số chiều dữ liệu. Nếu giải trực tiếp bằng các công cụ giải quy hoạch toàn phương, bài toán đối ngẫu có thể phức tạp hơn bài toán gốc. Tuy nhiên, điểm hấp dẫn của bài toán đối ngẫu này đến từ cấu trúc đặc biệt của hệ điều kiện KKT.

### 26.3.5. Điều kiện KKT

Quay trở lại bài toán, vì đây là một bài toán tối ưu lồi và đối ngẫu mạnh xảy ra, nghiệm của bài toán thoả mãn hệ điều kiện KKT sau đây với biến số w, b và λ:

<span id="page-7-0"></span>
$$1 - y_n(\mathbf{w}^T \mathbf{x}_n + b) \le 0, \ \forall n = 1, 2, \dots, N$$
 (26.12)

$$\lambda_n \ge 0, \ \forall n = 1, 2, \dots, N$$
 (26.13)

$$\lambda_n(1 - y_n(\mathbf{w}^T \mathbf{x}_n + b)) = 0, \ \forall n = 1, 2, \dots, N$$
 (26.14)

$$\mathbf{w} = \sum_{n=1}^{N} \lambda_n y_n \mathbf{x}_n \tag{26.15}$$

$$\sum_{n=1}^{N} \lambda_n y_n = 0 \tag{26.16}$$

Trong những điều kiện trên, điều kiện lỏng lẻo bù trừ [\(26.14\)](#page-7-0) là thú vị nhất. Từ đó ta có thể suy ra λ<sup>n</sup> = 0 hoặc 1 − yn(w<sup>T</sup> x<sup>n</sup> + b) = 0 với n bất kỳ. Trường hợp thứ hai tương đương với

<span id="page-7-1"></span>
$$\mathbf{w}^T \mathbf{x}_n + b = y_n. \tag{26.17}$$

Những điểm thoả mãn [\(26.17\)](#page-7-1) chính là những điểm nằm gần mặt phân chia nhất (những điểm được khoanh tròn trong Hình [26.3\)](#page-3-0). Hai đường thẳng w<sup>T</sup> xn+b = ±1 tựa lên các vector thoả mãn [\(26.17\)](#page-7-1). Những vector thoả mãn [\(26.17\)](#page-7-1) được gọi là vector hỗ trợ(support vector). Tên gọi máy vector hỗ trợ xuất phát từ đây.

Số lượng điểm thoả mãn [\(26.17\)](#page-7-1) thường chiếm một lượng nhỏ trong số N điểm dữ liệu huấn luyện. Chỉ cần dựa trên những vector hỗ trợ này, chúng ta hoàn toàn có thể xác định được mặt phân cách cần tìm. Nói cách khác, hầu hết các λ<sup>n</sup> bằng không, tức λ là một vector thưa. Máy vector hỗ trợ vì vậy cũng được coi là một mô hình thưa (sparse model). Các mô hình thưa thường có cách giải quyết hiệu quả hơn các mô hình tương tự với nghiệm dày đặc (dense, hầu hết các phần tử khác không). Đây là lý do thứ hai của việc bài toán đối ngẫu SVM được quan tâm nhiều hơn là bài toán chính.

Tiếp tục phân tích, với những bài toán với số điểm dữ liệu N nhỏ, ta có thể giải hệ điều kiện KKT phía trên bằng cách xét các trường hợp  $\lambda_n=0$  hoặc  $\lambda_n\neq 0$ . Tổng số trường hợp phải xét là  $2^N$ . Thông thường, N>50 và  $2^N$  là một con số rất lớn. Việc thử  $2^N$  trường hợp là bất khả thi. Phương pháp thường được dùng để giải hệ này là sequential minimal optimization (SMO) [Pla98, ZYX<sup>+</sup>08]. Trong phạm vi cuốn sách, chúng ta sẽ không đi sâu tiếp vào việc giải hệ KKT như thế nào.

Trong phần tiếp theo chúng ta sẽ giải bài toán tối ưu (26.11) qua một ví dụ nhỏ bằng CVXOPT, và trực tiếp sử dụng thư viện **sklearn** để huấn luyện mô hình SVM. Sau khi tìm được  $\lambda$  từ bài toán (26.11), ta có thể suy ra  $\mathbf{w}$  dựa vào (26.15) và b dựa vào (26.14) và (26.16). Rõ ràng ta chỉ cần quan tâm tới  $\lambda_n \neq 0$ .

Đặt  $S = \{n : \lambda_n \neq 0\}$  và  $N_S$  là số phần tử của S. Theo (26.15),  $\mathbf{w}$  được tính bằng

<span id="page-8-2"></span>
$$\mathbf{w} = \sum_{m \in \mathcal{S}} \lambda_m y_m \mathbf{x}_m. \tag{26.18}$$

Với mỗi  $n \in \mathcal{S}$ , ta có

$$1 = y_n(\mathbf{w}^T \mathbf{x}_n + b) \Leftrightarrow b = y_n - \mathbf{w}^T \mathbf{x}_n.$$

Mặc dù hoàn toàn có thể suy ra b từ một cặp  $(\mathbf{x}_n, y_n)$  nếu đã biết  $\mathbf{w}$ , một phiên bản tính b khác thường được sử dụng và có phần ổn định hơn trong tính toán là trung bình cộng<sup>65</sup> của các b tính được theo mỗi  $n \in \mathcal{S}$ 

<span id="page-8-1"></span>
$$b = \frac{1}{N_{\mathcal{S}}} \sum_{n \in \mathcal{S}} (y_n - \mathbf{w}^T \mathbf{x}_n) = \frac{1}{N_{\mathcal{S}}} \sum_{n \in \mathcal{S}} \left( y_n - \sum_{m \in \mathcal{S}} \lambda_m y_m \mathbf{x}_m^T \mathbf{x}_n \right)$$
(26.19)

Để xác định một điểm  ${\bf x}$  thuộc vào lớp nào, ta cần tìm dấu của biểu thức

$$\mathbf{w}^{T}\mathbf{x} + b = \sum_{m \in \mathcal{S}} \lambda_{m} y_{m} \mathbf{x}_{m}^{T} \mathbf{x} + \frac{1}{N_{\mathcal{S}}} \sum_{n \in \mathcal{S}} \left( y_{n} - \sum_{m \in \mathcal{S}} \lambda_{m} y_{m} \mathbf{x}_{m}^{T} \mathbf{x}_{n} \right).$$

Biểu thức này phụ thuộc vào cách tính tích vô hướng giữa  $\mathbf{x}$  và từng  $\mathbf{x}_m \in \mathcal{S}$ . Nhận xét quan trọng này sẽ giúp ích cho chúng ta trong chương 28.

## 26.4. Lập trình tìm nghiệm cho máy vector hỗ trợ

Trong mục này, ta sẽ tìm nghiệm của SVM bằng hai cách khác nhau. Cách thứ nhất dựa trên bài toán (26.11) với nghiệm tìm được theo các công thức (26.19) và (26.18). Cách làm này giúp chứng minh tính đúng đắn của các công thức đã xây dựng. Cách thứ hai sử dụng trực tiếp thư viện sklearn, giúp bạn đọc làm quen với việc áp dụng SVM vào dữ liệu thực tế.

<span id="page-8-0"></span> $<sup>^{65}</sup>$  Việc lấy trung bình này giống cách đo trong các thí nghiệm vật lý. Để đo một đại lượng, người ta thường thực hiện việc đo nhiều lần rồi lấy kết quả trung bình để tránh sai số. Ở đây, về mặt toán học, b phải như nhau theo mọi cách tính. Tuy nhiên, khi tính toán bằng máy tính, chúng ta có thể gặp các sai số nhỏ. Việc lấy trung bình sẽ làm giảm sai số đó.

### 26.4.1. Tìm nghiệm theo công thức

Trước tiên ta khai báo các thư viện và tạo dữ liệu giả (dữ liệu này được sử dụng trong các hình từ đầu chương. Ta thấy rằng hai lớp dữ liệu tách biệt tuyến tính):

```
from __future__ import print_function
import numpy as np
np.random.seed(22)
# simulated samples
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # blue class data
X1 = np.random.multivariate_normal(means[1], cov, N) # red class data
X = np.concatenate((X0, X1), axis = 0) # all data
y = np.concatenate((np.ones(N), -np.ones(N)), axis = 0) # label
# solving the dual problem (variable: lambda)
from cvxopt import matrix, solvers
V = np.concatenate((X0, -X1), axis = 0) # V in the book
Q = matrix(V.dot(V.T))
p = matrix(-np.ones((2*N, 1))) # objective function 1/2 lambda^T*Q*lambda -
   1^T*lambda
# build A, b, G, h
G = matrix(-np.eye(2*N))
h = matrix(np.zeros((2*N, 1)))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros((1, 1)))
solvers.options['show_progress'] = False
sol = solvers.qp(Q, p, G, h, A, b)
l = np.array(sol['x']) # solution lambda
# calculate w and b
w = Xbar.T.dot(l)
S = np.where(l > 1e-8)[0] # support set, 1e-8 to avoid small value of l.
b = np.mean(y[S].reshape(-1, 1) - X[S,:].dot(w))
print('Number of suport vectors = ', S.size)
print('w = ', w.T)
print('b = ', b)
```

Kết quả:

```
Number of suport vectors = 3
w = [[-2.00984382 0.64068336]]
b = 4.66856068329
```

Như vậy trong số 20 điểm dữ liệu của cả hai lớp, chỉ có ba điểm đóng vai trò vector hỗ trợ. Ba điểm này giúp tính w và b. Đường thẳng phân chia tìm được có màu đen đậm và được minh hoạ trong Hình [26.4.](#page-10-0) Hai đường đen mảnh thể hiện đường thẳng tựa lên các vector hỗ trợ được khoanh tròn.

Hình vẽ và mã nguồn trong bài có thể được tìm thấy tại <https://goo.gl/VKBgVG>.

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

Hình 26.4. Minh hoạ nghiệm tìm được bởi SVM. Tất cả các điểm nằm trong vùng có nền kẻ ô sẽ được phân vào cùng lớp với các điểm vuông. Điều tương tự xảy ra với các điểm tròn nằm trên nền dấu chấm.

### 26.4.2. Tìm nghiệm theo thư viện

Chúng ta sẽ sử dụng [sklearn.svm.SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[66](#page-10-1). Bạn đọc có thể tham khảo thêm thư viện [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) được viết trên ngôn ngữ C, có API cho Python và Matlab:

```
# solution by sklearn
from sklearn.svm import SVC
model = SVC(kernel = 'linear', C = 1e5) # just a big number
model.fit(X, y)
w = model.coef_
b = model.intercept_
print('w = ', w)
print('b = ', b)
```

Kết quả:

```
w = [[-2.00971102 0.64194082]]
b = [ 4.66595309]
```

Kết quả này thống nhất với kết quả tìm được ở mục trước. Có rất nhiều tuỳ chọn cho SVC, trong đó có thuộc tính kernel, các bạn sẽ dần thấy trong các chương sau.

## 26.5. Tóm tắt

- Nếu hai lớp dữ liệu tách biệt tuyến tính, có vô số các siêu phẳng phân chia hai lớp đó. Khoảng cách gần nhất từ một điểm dữ liệu tới siêu phẳng này được gọi là lề.
- SVM là bài toán đi tìm mặt phân cách sao cho lề của hai lớp bằng nhau và lớn nhất, đồng nghĩa với việc các điểm dữ liệu có một khoảng cách an toàn tới mặt phân chia.

<span id="page-10-1"></span><sup>66</sup> SVC là viết tắt của bộ phân loại vector hỗ trợ (support vector classifier).

- Bài toán tối ưu trong SVM là một bài toán quy hoạch toàn phương với hàm mục tiêu lồi chặt. Vì vậy, cực tiểu địa phương cũng là cực tiểu toàn cục của bài toán.
- Mặc dù có thể trực tiếp giải SVM qua bài toán chính, người ta thường giải bài toán đối ngẫu. Bài toán đối ngẫu cũng là một bài toán quy hoạch toàn phương nhưng nghiệm là các vector thưa nên có những phương pháp giải hiệu quả hơn. Ngoài ra, bài toán đối ngẫu có những tính chất thú vị sẽ được thảo luận trong các chương tiếp theo.