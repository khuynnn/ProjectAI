# Giảm chiều dữ liệu

Các bài toán quy mô lớn trên thực tế có lượng điểm dữ liệu lớn và dữ liệu nhiều chiều. Nếu thực hiện lưu trữ và tính toán trực tiếp trên dữ liệu có số chiều lớn thì sẽ gặp khó khăn về lưu trữ và tính toán. Vì vậy,  $giảm\ chiều\ dữ\ liệu\ (dimensionality\ reduction\ hoặc\ dimension\ reduction)$  là một bước quan trọng trong nhiều bài toán machine learning.

Dưới góc độ toán học, giảm chiều dữ liệu là việc đi tìm một hàm số  $f:\mathbb{R}^D\to\mathbb{R}^K$  với K< D biến một điểm dữ liệu  $\mathbf{x}$  trong không gian có số chiều lớn  $\mathbb{R}^D$  thành một điểm  $\mathbf{z}$  trong không gian có số chiều nhỏ hơn  $\mathbb{R}^D$ . Giảm chiều dữ liệu có thể áp dụng vào các bài toán nén thông tin. Nó cũng hữu ích trong việc chọn ra những đặc trưng quan trọng hoặc tạo ra các đặc trưng mới từ đặc trưng cũ phù hợp với từng bài toán. Trong nhiều trường hợp, làm việc trên dữ liệu đã giảm chiều cho kết quả tốt hơn dữ liệu trong không gian ban đầu.

Trong phần này, chúng ta sẽ xem xét các phương pháp giảm chiều dữ liệu phổ biến nhất: phân tích thành phần chính (principle component analysis) cho bài toán giảm chiều dữ liệu vẫn giữ tối đa lượng thông tin, và linear discriminant analysis cho bài toán giữ lại những đặc trưng quan trọng nhất cho việc phân loại. Trước hết, chúng ta cùng tìm hiểu một phương pháp phân tích ma trận vô cùng quan trọng -phân tích giá tri suy biến (singular value decomposition).

# Phân tích giá trị suy biến

# 20.1. Giới thiệu

Nhắc lại bài toán chéo hoá ma trận: Một ma trận vuông A ∈ R <sup>n</sup>×<sup>n</sup> gọi là chéo hoá được nếu tồn tại ma trận đường chéo D và ma trận khả nghịch P sao cho:

<span id="page-1-0"></span>
$$\mathbf{A} = \mathbf{P}\mathbf{D}\mathbf{P}^{-1} \tag{20.1}$$

Nhân cả hai vế của [\(20.1\)](#page-1-0) với P ta có

<span id="page-1-1"></span>
$$\mathbf{AP} = \mathbf{PD} \tag{20.2}$$

Gọi p<sup>i</sup> , d<sup>i</sup> lần lượt là cột thứ i của ma trận P và D. Vì mỗi cột của vế trái và vế phải của [\(20.2\)](#page-1-1) phải bằng nhau, ta cần có

<span id="page-1-2"></span>
$$\mathbf{A}\mathbf{p}_i = \mathbf{P}\mathbf{d}_i = d_{ii}\mathbf{p}_i \tag{20.3}$$

với dii là phần tử thứ i của d<sup>i</sup> . Dấu bằng thứ hai xảy ra vì D là ma trận đường chéo, tức d<sup>i</sup> chỉ có thành phần dii khác không. Biểu thức [\(20.3\)](#page-1-2) chỉ ra rằng mỗi phần tử dii phải là một trị riêng của A và mỗi vector cột p<sup>i</sup> phải là một vector riêng của A ứng với trị riêng dii.

Cách phân tích một ma trận vuông thành nhân tử như [\(20.1\)](#page-1-0) còn được gọi là phân tích riêng (eigen decomposition). Đáng chú ý, không phải lúc nào cũng tồn tại cách phân tích này cho một ma trận bất kỳ. Nó chỉ tồn tại nếu ma trận A có n vector riêng độc lập tuyến tính, tức ma trận P khả nghịch. Thêm nữa, cách phân tích này không phải là duy nhất vì nếu P, D thoả mãn [\(20.1\)](#page-1-0) thì kP, D cũng thoả mãn với k là một số thực khác không bất kỳ.

Việc phân tích một ma trận thành tích của nhiều ma trận đặc biệt khác mang lại những ích lợi quan trọng trong bài toán gợi ý sản phẩm, giảm chiều dữ liệu, nén dữ liệu, tìm hiểu các đặc tính của dữ liệu, giải các hệ phương trình tuyến tính, phân cụm và nhiều ứng dụng khác.

Trong chương này, chúng ta sẽ làm quen với một trong những phương pháp phân tích ma trận rất đẹp của đại số tuyến tính có tên là phân tích giá trị suy biên (singular value decomposition – SVD) [GR70]. Mọi ma trận, không nhất thiết vuông, đều có thể được phân tích thành tích của ba ma trận đặc biệt.

# 20.2. Phân tích giá trị suy biến

Để hạn chế nhầm lẫn trong các phép toán, ta sẽ ký hiệu một ma trận cùng với kích thước của nó, ví dụ A<sup>m</sup>×<sup>n</sup> ký hiệu một ma trận A ∈ R m×n .

## 20.2.1. Phát biểu phân tích giá trị suy biến

#### Phân tích giá trị suy biến (SVD)

Một ma trận A<sup>m</sup>×<sup>n</sup> bất kỳ đều có thể phân tích thành dạng:

<span id="page-2-1"></span>
$$\mathbf{A}_{m \times n} = \mathbf{U}_{m \times m} \mathbf{\Sigma}_{m \times n} (\mathbf{V}_{n \times n})^T$$
 (20.4)

với U, V là các ma trận trực giao, Σ là một ma trận đường chéo cùng kích thước với A. Các phần tử trên đường chéo chính của Σ là không âm và được sắp xếp theo thứ tự giảm dần σ<sup>1</sup> ≥ σ<sup>2</sup> ≥ · · · ≥ σ<sup>r</sup> ≥ 0 = 0 = · · · = 0. Số lượng các phần tử khác không trong Σ chính là hạng của ma trận A: r = rank(A).

SVD của một ma trận bất kỳ luôn tồn tại[51](#page-2-0). Cách biểu diễn [\(20.4\)](#page-2-1) không là duy nhất vì ta chỉ cần đổi dấu của cả U và V thì [\(20.4\)](#page-2-1) vẫn thoả mãn.

Hình [20.1](#page-3-0) mô tả SVD của ma trận A<sup>m</sup>×<sup>n</sup> trong hai trường hợp: m < n và m > n. Trường hợp m = n có thể xếp vào một trong hai trường hợp trên.

# 20.2.2. Nguồn gốc tên gọi

Tạm bỏ qua chiều của mỗi ma trận, từ [\(20.4\)](#page-2-1) ta có:

<span id="page-2-2"></span>
$$\mathbf{A}\mathbf{A}^T = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T)^T \tag{20.5}$$

$$= \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \mathbf{V} \mathbf{\Sigma}^T \mathbf{U}^T \tag{20.6}$$

$$= \mathbf{U} \mathbf{\Sigma} \mathbf{\Sigma}^T \mathbf{U}^T = \mathbf{U} \mathbf{\Sigma} \mathbf{\Sigma}^T \mathbf{U}^{-1}$$
 (20.7)

Dấu bằng ở [\(20.6\)](#page-2-2) xảy ra vì V<sup>T</sup>V = I do V là một ma trận trực giao. Dấu bằng ở [\(20.7\)](#page-2-2) xảy ra vì U là một ma trận trực giao.

<span id="page-2-0"></span><sup>51</sup> Bạn đọc có thể tìm thấy chứng minh cho việc này tại <https://goo.gl/TdtWDQ>.

<span id="page-3-0"></span>![](_page_3_Picture_1.jpeg)

**Hình 20.1.** SVD cho ma trận  $\bf A$  khi: (a) m < n, và (b) m > n.  $\Sigma$  là một ma trận đường chéo với các phần tử trên đó giảm dần và không âm. Màu xám càng đậm thể hiện giá trị càng cao. Các ô màu trắng trên ma trận  $\Sigma$  thể hiện giá trị bằng không.

Quan sát thấy rằng  $\Sigma\Sigma^T$  là một ma trận đường chéo với các phần tử trên đường chéo là  $\sigma_1^2, \sigma_2^2, \ldots$  Vậy (20.7) chính là một phân tích riêng của  $\mathbf{A}\mathbf{A}^T$  và  $\sigma_1^2, \sigma_2^2, \ldots$  là các trị riêng của ma trận này. Ma trận  $\mathbf{A}\mathbf{A}^T$  luôn là nửa xác định dương nên các trị riêng của nó là không âm. Căn bậc hai các trị riêng của  $\mathbf{A}\mathbf{A}^T$ ,  $\sigma_i$ , còn được gọi là giá trị suy biến (singular value) của  $\mathbf{A}$ . Tên gọi phân tích giá trị suy biến xuất phát từ đây.

Cũng theo đó, mỗi cột của  $\mathbf{U}$  là một vector riêng của  $\mathbf{A}\mathbf{A}^T$ . Ta gọi mỗi cột này là một vector suy biến trái (left-singular vector) của  $\mathbf{A}$ . Tương tự,  $\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{V}^T$  và các cột của  $\mathbf{V}$  được gọi là các vector suy biến phải (right-singular vectors) của  $\mathbf{A}$ .

Trong Python, để tính SVD của một ma trận, chúng ta sử dụng module linal<br/>g của numpy:

```
from __future__ import print_function\nimport numpy as np
from numpy import linalg as LA

m, n = 3, 4
A = np.random.rand(m, n)
U, S, V = LA.svd(A) # A = U*S*V (no V transpose here)

# checking if U, V are orthogonal and S is a diagonal matrix with
# nonnegative decreasing elements
print('Frobenius norm of (UU^T - I) =', LA.norm(U.dot(U.T) - np.eye(m)))
print('S = ', S)
print('Frobenius norm of (VV^T - I) =', LA.norm(V.dot(V.T) - np.eye(n)))
```

<span id="page-4-0"></span>![](_page_4_Picture_1.jpeg)

Hình 20.2. Biểu diễn compact SVD dưới dạng tổng các ma trận có rank bằng 1. Các khối ma trận đặt cạnh nhau thể hiện phép nhân ma trận.

#### Kết quả:

```
Frobenius norm of (UU^T - I) = 4.09460889695e-16
S = [ 1.76321041 0.59018069 0.3878011 ]
Frobenius norm of (VV^T - I) = 5.00370755311e-16
```

Lưu ý rằng biến S được trả về chỉ bao gồm các phần tử trên đường chéo của Σ. Biến V trả về là V<sup>T</sup> trong [\(20.4\)](#page-2-1).

#### 20.2.3. Giá trị suy biến của ma trận nửa xác định dương

Giả sử A là một ma trận vuông đối xứng nửa xác định dương, ta sẽ chứng minh rằng giá trị suy biến chính là trị riêng của nó. Thật vậy, gọi λ là một trị riêng của A và x là một vector riêng ứng với trị riêng và kxk<sup>2</sup> = 1. Vì A là nửa xác định dương, λ ≥ 0. Ta có

$$\mathbf{A}\mathbf{x} = \lambda \mathbf{x} \Rightarrow \mathbf{A}^T \mathbf{A}\mathbf{x} = \lambda \mathbf{A}\mathbf{x} = \lambda^2 \mathbf{x}$$
 (20.8)

Như vậy, λ 2 là một trị riêng của <sup>A</sup><sup>T</sup><sup>A</sup> <sup>⇒</sup> giá trị suy biến của <sup>A</sup> chính là <sup>√</sup> λ <sup>2</sup> = λ.

# 20.2.4. Phân tích giá trị suy biến giản lược

Viết lại biểu thức [\(20.4\)](#page-2-1) dưới dạng tổng của các ma trận có hạng bằng môt:

$$\mathbf{A} = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \dots + \sigma_r \mathbf{u}_r \mathbf{v}_r^T$$
 (20.9)

với chú ý rằng mỗi uiv T i , 1 ≤ i ≤ r, là một ma trận có hạng bằng một.

Trong cách biểu diễn này, ma trận A chỉ phụ thuộc vào r cột đầu tiên của U, V và r giá trị khác 0 trên đường chéo của ma trận Σ. Vì vậy ta có một cách phân tích gọn hơn và gọi là SVD giản lược (compact SVD):

$$\mathbf{A} = \mathbf{U}_r \mathbf{\Sigma}_r (\mathbf{V}_r)^T \tag{20.10}$$

với Ur, V<sup>r</sup> lần lượt là ma trận được tạo bởi r cột đầu tiên của U và V. Σ<sup>r</sup> là ma trận con được tạo bởi r hàng đầu tiên và r cột đầu tiên của Σ. Nếu ma trận A có hạng nhỏ hơn rất nhiều so với số hàng và số cột, tức r m, n, ta sẽ được lợi nhiều về việc lưu trữ. Hình [20.2](#page-4-0) là một ví dụ minh hoạ với m = 4, n = 6, r = 2.

#### 20.2.5. Phân tích giá tri suy biến cắt ngọn

Nhắc lại rằng các giá trị trên đường chéo chính của  $\Sigma$  là không âm và giảm dần  $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0 = 0 = \cdots = 0$ . Thông thường, chỉ một lượng nhỏ các  $\sigma_i$  mang giá trị lớn, các giá trị còn lại nhỏ và gần không. Khi đó ta có thể xấp xỉ ma trận  $\mathbf{A}$  bằng tổng của k < r ma trận có hạng bằng một:

$$\mathbf{A} \approx \mathbf{A}_k = \mathbf{U}_k \mathbf{\Sigma}_k (\mathbf{V}_k)^T = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^T + \dots + \sigma_k \mathbf{u}_k \mathbf{v}_k^T$$
(20.11)

Việc bỏ đi r-k giá trị suy biến khác không nhỏ nhất được gọi là SVD cắt ngọn (truncated SVD). Dưới đây là một định lý thú vị. Định lý này nói rằng sai số do cách xấp xỉ SVD cắt ngọn bằng căn bậc hai tổng bình phương của các giá trị suy biến bị cắt đi. Ở đây sai số được định nghĩa là Frobineous norm của hiệu hai ma trân.

#### Định lý 20.1: Sai số do xấp xỉ bởi SVD cắt ngọn

Sai số do xấp xỉ một ma trận  ${\bf A}$  có hạng r bởi SVD cắt ngọn với k < r phần tử là

$$\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$
 (20.12)

Chứng minh: Sử dụng tính chất  $\|\mathbf{X}\|_F^2 = \operatorname{trace}(\mathbf{X}\mathbf{X}^T)$  và  $\operatorname{trace}(\mathbf{X}\mathbf{Y}) = \operatorname{trace}(\mathbf{Y}\mathbf{X})$  với mọi ma trận  $\mathbf{X}, \mathbf{Y}$  ta có:

<span id="page-5-0"></span>
$$\|\mathbf{A} - \mathbf{A}_{k}\|_{F}^{2} = \left\| \sum_{i=k+1}^{r} \sigma_{i} \mathbf{u}_{i} \mathbf{v}_{i}^{T} \right\|_{F}^{2} = \operatorname{trace} \left\{ \left( \sum_{i=k+1}^{r} \sigma_{i} \mathbf{u}_{i} \mathbf{v}_{i}^{T} \right) \left( \sum_{j=k+1}^{r} \sigma_{j} \mathbf{u}_{j} \mathbf{v}_{j}^{T} \right)^{T} \right\}$$

$$= \operatorname{trace} \left\{ \sum_{i=k+1}^{r} \sum_{j=k+1}^{r} \sigma_{i} \sigma_{j} \mathbf{u}_{i} \mathbf{v}_{i}^{T} \mathbf{v}_{j} \mathbf{u}_{j}^{T} \right\} = \operatorname{trace} \left\{ \sum_{i=k+1}^{r} \sigma_{i}^{2} \mathbf{u}_{i} \mathbf{u}_{i}^{T} \right\} (20.13)$$

$$= \operatorname{trace} \left\{ \sum_{i=k+1}^{r} \sigma_{i}^{2} \mathbf{u}_{i}^{T} \mathbf{u}_{i} \right\} (20.14)$$

$$= \operatorname{trace} \left\{ \sum_{i=k+1}^{r} \sigma_{i}^{2} \right\} = \sum_{i=k+1}^{r} \sigma_{i}^{2} (20.15)$$

Dấu bằng thứ hai ở (20.13) xảy ra vì  $\mathbf{V}$  có các cột vuông góc với nhau. Dấu bằng ở (20.14) xảy ra vì hàm trace có tính chất giao hoán. Dấu bằng ở (20.15) xảy ra vì biểu thức trong dấu ngoặc là một số vô hướng.

Thay k=0 ta sẽ có

$$\|\mathbf{A}\|_F^2 = \sum_{i=1}^r \sigma_i^2 \tag{20.16}$$

Từ đó

$$\frac{\|\mathbf{A} - \mathbf{A}_k\|_F^2}{\|\mathbf{A}\|_F^2} = \frac{\sum_{i=k+1}^r \sigma_i^2}{\sum_{j=1}^r \sigma_j^2}$$
(20.17)

Như vậy, sai số do xấp xỉ càng nhỏ nếu các giá trị suy biến bị cắt càng nhỏ so với các giá trị suy biến được giữ lại. Đây là một định lý quan trọng giúp xác định việc xấp xỉ ma trận dựa trên lượng thông tin muốn giữ lại. Ở đây, lượng thông tin được định nghĩa là tổng bình phương của giá trị suy biến. Ví dụ, nếu muốn giữ lại ít nhất 90% lượng thông tin trong  $\mathbf{A}$ , trước hết ta tính  $\sum_{j=1}^{r} \sigma_{j}^{2}$ , sau đó chọn k là số nhỏ nhất sao cho

$$\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{j=1}^{r} \sigma_j^2} \ge 0.9 \tag{20.18}$$

Khi k nhỏ, ma trận  $\mathbf{A}_k$  có hạng nhỏ bằng k Vì vậy, SVD cắt ngọn cũng được xếp vào loại  $x \hat{a} p \ x \hat{i} \ hạng \ th \hat{a} p$ .

#### 20.2.6. Xấp xỉ hạng k tốt nhất

Người ta chứng minh được rằng 52  $\mathbf{A}_k$  chính là nghiệm của bài toán tối ưu sau đây:

$$\min_{\mathbf{B}} \|\mathbf{A} - \mathbf{B}\|_{F}$$
thoả mãn: rank( $\mathbf{B}$ ) =  $k$  (20.19)

và  $\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$  như đã chứng minh ở trên .

Nếu sử dụng  $\ell_2$  norm của ma trận (xem Phụ lục A) thay vì Frobenius norm để đo sai số,  $\mathbf{A}_k$  cũng là nghiệm của bài toán tối ưu

$$\min_{\mathbf{B}} \|\mathbf{A} - \mathbf{B}\|_{2}$$
thoả mãn: rank( $\mathbf{B}$ ) =  $k$  (20.20)

và sai số  $\|\mathbf{A} - \mathbf{A}_k\|_2^2 = \sigma_{k+1}^2$ . Trong đó,  $\ell_2$  norm của một ma trận được định nghĩa bởi

$$\|\mathbf{A}\|_{2} = \max_{\|\mathbf{x}\|_{2}=1} \|\mathbf{A}\mathbf{x}\|_{2}$$
 (20.21)

Frobenius norm và  $\ell_2$  norm là hai norm được sử dụng nhiều nhất trong ma trận. Như vậy, xét trên cả hai norm này, SVD cắt ngọn đều cho xấp xỉ tốt nhất. Vì vậy, SVD cắt ngọn còn được coi là  $x \hat{a} p$   $x \hat{i}$  hạng thấp tốt nhất (best low-rank approximation).

# 20.3. Phân tích giá trị suy biến cho bài toán nén ảnh

Xét ví dụ trong Hình 20.3. Bức ảnh gốc trong Hình 20.3<br/>a là một ảnh xám có kích thước  $960 \times 1440$  điểm ảnh. Bức ảnh này có thể được coi là một ma trận

<span id="page-6-0"></span><sup>52</sup> Singular Value Decomposition – Princeton (https://goo.gl/hU38GF).

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

**Hình 20.3.** Ví dụ về SVD cho ảnh. (a) Bức ảnh gốc là một ma trận cỡ  $960 \times 1440$ . (b) Các giá trị suy biến của ma trận ảnh theo thang đo logarit. Các giá trị suy biến giảm nhanh ở khoảng k=200. (c) Biểu diễn lượng thông tin được giữ lại khi chọn các k khác nhau. Có thể thấy từ khoảng k=200, lượng thông tin giữ lại gần bằng 1. Vậy ta có thể xấp xỉ ma trận ảnh này bằng một ma trận có hạng nhỏ hơn. (d), (e), (f) Các ảnh xấp xỉ với k lần lượt là 5, 50, 100.

 $\mathbf{A} \in \mathbb{R}^{960 \times 1440}$ . Có thể thấy rằng ma trận này có hạng thấp vì toà nhà có các tầng tương tự nhau, tức ma trận có nhiều hàng tương tự nhau. Hình 20.3b thể hiện các giá trị suy biến sắp xếp theo thứ tự giảm dần của ma trận điểm ảnh. Ở đây, các giá trị suy biến được biểu diễn trong thang logarit thập phân. Giá trị suy biến đầu tiên lớn hơn giá trị suy biến thứ 250 khoảng gần 1000 lần. Hình 20.3c mô tả chất lượng của việc xấp xỉ  $\mathbf{A}$  bởi  $\mathbf{A}_k$  thông qua SVD cắt ngọn. Ta thấy giá trị này xấp xỉ bằng một tại k=200. Hình 20.3d, 20.3e, 20.3f là các bức ảnh xấp xỉ khi chọn các giá trị k khác nhau. Khi k gần 100, lượng thông tin mất đi hơn 0.3%, ảnh thu được có chất lượng gần như ảnh gốc.

Để lưu ảnh với SVD cắt ngọn, ta lưu các ma trận  $\mathbf{U}_k \in \mathbb{R}^{m \times k}, \mathbf{\Sigma}_k \in \mathbb{R}^{k \times k}, \mathbf{V}_k \in \mathbb{R}^{n \times k}$ . Tổng số phần tử phải lưu là k(m+n+1) với chú ý rằng  $\mathbf{\Sigma}_k$  là một ma trận đường chéo. Nếu mỗi phần tử được lưu bởi một số thực bốn byte thì số byte cần lưu là 4k(m+n+1). Nếu so giá trị này với ảnh gốc có kích thước mn, mỗi giá trị là một số nguyên một byte, tỉ lệ nén là

$$\frac{4k(m+n+1)}{mn} \tag{20.22}$$

Khi  $k \ll m, n$ , ta được một tỉ lệ nhỏ hơn 1. Trong ví dụ trên, m = 960, n = 1440, k = 100, tỉ lệ nén là xấp xỉ 0.69, tức đã tiết kiệm được khoảng 30% bộ nhớ.

# 20.4. Thảo luận

- Ngoài những ứng dụng nêu trên, SVD còn được áp dụng trong việc giải phương trình tuyến tính thông qua giả nghịch đảo Moore Penrose ([https://goo.gl/](https://goo.gl/4wrXue) [4wrXue](https://goo.gl/4wrXue)), hệ thống gợi ý [SKKR00], giảm chiều dữ liệu [Cyb89], khử mờ ảnh (image deblurring) [HNO06], phân cụm [DFK<sup>+</sup>04],...
- Khi ma trận A lớn, việc tính toán SVD tốn nhiều thời gian. Cách tính SVD cắt ngọn với k như được trình bày trở nên không khả thi. Có một phương pháp lặp giúp tính các trị riêng và vector riêng của một ma trận lớn một cách hiệu quả. Trong phương pháp này, ta chỉ cần tìm k trị riêng lớn nhất của AA<sup>T</sup> và các vector riêng tương ứng. Việc này giúp khối lượng tính toán giảm đi đáng kể. Bạn đọc có thể tìm đọc thêm Power method for approximating eigenvalues (<https://goo.gl/PfDqsn>).
- Mã nguồn trong chương này có thể được tìm thấy tại <https://goo.gl/Z3wbsU>.

#### Đọc thêm

- a. Singular Value Decomposition Stanford University (<https://goo.gl/Gp726X>).
- b. Singular Value Decomposition Princeton (<https://goo.gl/HKpcsB>).
- c. CS168: The Modern Algorithmic Toolbox Lecture #9: The Singular Value Decomposition (SVD) and Low-Rank Matrix Approximations - Stanford ([https:](https://goo.gl/RV57KU) [//goo.gl/RV57KU](https://goo.gl/RV57KU)).
- d. The Moore-Penrose Pseudoinverse (Math 33A UCLA) (<https://goo.gl/VxMYx1>).