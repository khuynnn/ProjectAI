# Đối ngẫu

## 25.1. Giới thiệu

Trong Chương 23 và Chương 24, chúng ta đã thảo luận về tập lồi, hàm lồi và các bài toán tối ưu lồi. Trong chương này, chúng ta sẽ tiếp tục tìm hiểu sâu hơn scác điều kiện về nghiệm của các bài toán tối ưu, cả lồi và không lồi; bài toán đối ngẫu (dual problem) và điều kiện KKT.

Trước tiên chúng ta xét bài toán tối ưu chỉ có một phương trình ràng buộc:

<span id="page-0-0"></span>
$$\mathbf{x} = \arg\min_{\mathbf{x}} f_0(\mathbf{x})$$
thoả mãn:  $f_1(\mathbf{x}) = 0$  (25.1)

Bài toán này không nhất thiết là bài toán tối ưu lồi. Tức hàm mục tiêu và hàm ràng buộc không nhất thiết phải lồi. Bài toán này có thể được giải bằng phương pháp nhân tử Lagrange (xem Phụ Lục A). Cụ thể, xét hàm số:

$$\mathcal{L}(\mathbf{x},\lambda) = f_0(\mathbf{x}) + \lambda f_1(\mathbf{x})$$
 (25.2)

Hàm số L(x, λ) được gọi là hàm Lagrange (the Lagrangian) của bài toán tối ưu [\(25.1\)](#page-0-0). Trong hàm số này, chúng ta có thêm một biến λ được gọi là nhân tử Lagrange (Lagrange multiplier). Người ta đã chứng minh được rằng, điểm tối ưu của bài toán [\(25.1\)](#page-0-0) thoả mãn điều kiện ∇x,λL(x, λ) = 0. Tức là:

<span id="page-0-1"></span>
$$\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda) = \nabla_{\mathbf{x}} f_0(\mathbf{x}) + \lambda \nabla_{\mathbf{x}} f_1(\mathbf{x}) = \mathbf{0}$$
 (25.3)

$$\nabla_{\lambda} \mathcal{L}(\mathbf{x}, \lambda) = f_1(\mathbf{x}) = 0 \tag{25.4}$$

Để ý rằng điều kiện thứ hai chính là phương trình ràng buộc trong bài toán [\(25.1\)](#page-0-0). Trong nhiều trường hợp, việc giải hệ phương trình [\(25.3\)](#page-0-1) - [\(25.4\)](#page-0-1) đơn giản hơn việc trực tiếp đi tìm optimal value của bài toán [\(25.1\)](#page-0-0). Một số ví dụ về phương pháp nhân tử Lagrange có thể được tìm thấy tại Phụ Lục A.

# 25.2. Hàm đối ngẫu Lagrange

#### 25.2.1. Hàm Lagrange của bài toán tối ưu

Xét bài toán tối ưu tổng quát:

<span id="page-1-0"></span>
$$\mathbf{x}^* = \arg\min_{\mathbf{x}} f_0(\mathbf{x})$$
thoả mãn:  $f_i(\mathbf{x}) \le 0, \quad i = 1, 2, \dots, m$  
$$h_j(\mathbf{x}) = 0, \quad j = 1, 2, \dots, p$$
 (25.5)

với tập xác định D = T<sup>m</sup> <sup>i</sup>=0 domfi) ∩ ( T<sup>p</sup> <sup>j</sup>=1 domh<sup>j</sup> . Chú ý rằng, ở đây không có giả sử về tính chất lồi của hàm tối ưu hay các hàm ràng buộc. Giả sử duy nhất là tập xác định D 6= ∅ (tập rỗng). Bài toán tối ưu này còn được gọi là bài toán chính (primal problem).

Hàm số Lagrange cũng được xây dựng tương tự với mỗi nhân tử Lagrange cho một (bất) phương trình ràng buộc:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x}).$$

Trong đó, λ = [λ1, λ2, . . . , λm]; ν = [ν1, ν2, . . . , νp] là các vector được gọi là biến đối ngẫu (dual variable) hoặc vector nhân tử Lagrange (Lagrange multiplier vector). Nếu biến chính x ∈ R n thì tổng số biến của hàm số Lagrange là n + m + p.

## 25.2.2. Hàm đối ngẫu Lagrange

Hàm đối ngẫu Lagrange (the Lagrange dual function) của bài toán tối ưu (viết gọn là hàm số đối ngẫu) [\(25.5\)](#page-1-0) là một hàm của các biến đối ngẫu λ và ν, được định nghĩa là infimum theo x của hàm Lagrange:

$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x} \in \mathcal{D}} \left( f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x}) \right)$$
(25.6)

Nếu hàm Lagrange không bị chặn dưới, hàm đối ngẫu tại λ, ν lấy giá trị −∞.

Lưu ý :

- inf được lấy trên miền x ∈ D, tức tập xác định của bài toán. Tập xác định này khác với tập khả thi – là tập hợp các điểm thoả mãn các ràng buộc.
- Với mỗi x, hàm số đối ngẫu là một hàm affine của (λ, ν), tức là một hàm vừa lồi, vừa lõm. Hàm đối ngẫu chính là một infimum từng thành phần của (có

thể vô hạn) các hàm lõm, tức cũng là một hàm lõm. Như vậy, hàm đối ngẫu của một bài toán tối ưu bất kỳ là một hàm lõm, bất kể bài toán tối ưu đó có là bài toán tối ưu lồi hay không. Nhắc lại rằng supremum từng thành phần của các hàm lồi là một hàm lồi; và một hàm là lõm nếu hàm đối của nó là một hàm lồi (xem thêm Mục 23.3.2).

#### 25.2.3. Chặn dưới của giá trị tối ưu

Nếu p ∗ là giá trị tối ưu của bài toán [\(25.5\)](#page-1-0) thì với các biến đối ngẫu λ<sup>i</sup> ≥ 0, ∀i và ν bất kỳ, ta sẽ có

<span id="page-2-0"></span>
$$g(\lambda, \nu) \le p^* \tag{25.7}$$

Tính chất này có thể được chứng minh như sau. Giả sử x<sup>0</sup> là một điểm khả thi bất kỳ của bài toán [\(25.5\)](#page-1-0), tức thoả mãn các điều kiện ràng buộc fi(x0) ≤ 0, ∀i = 1, . . . , m; h<sup>j</sup> (x0) = 0, ∀j = 1, . . . , p, ta sẽ có

$$\mathcal{L}(\mathbf{x}_0, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\mathbf{x}_0) + \sum_{i=1}^m \underbrace{\lambda_i f_i(\mathbf{x}_0)}_{\leq 0} + \sum_{j=1}^p \underbrace{\nu_j h_j(\mathbf{x}_0)}_{=0} \leq f_0(\mathbf{x}_0)$$

Vì điều này đúng với mọi điểm khả thi x0, ta có tính chất quan trọng sau đây:

$$g(\lambda, \nu) = \inf_{\mathbf{x} \in \mathcal{D}} \mathcal{L}(\mathbf{x}, \lambda, \nu) \le \mathcal{L}(\mathbf{x}_0, \lambda, \nu) \le f_0(\mathbf{x}_0).$$

Khi x<sup>0</sup> = x ∗ (điểm tối ưu), f0(x0) = p ∗ , ta suy ra bất đẳng thức [\(25.7\)](#page-2-0). Bất đẳng thức quan trọng này chỉ ra rằng giá trị tối ưu của hàm mục tiêu trong bài toán chính [\(25.5\)](#page-1-0) không nhỏ hơn giá trị lớn nhất của hàm đối ngẫu Lagrange g(λ, ν).

#### 25.2.4. Ví dụ

Ví dụ 1: Xét bài toán tối ưu:

$$x = \arg\min_{x} x^2 + 10\sin(x) + 10$$
thoả mãn:  $(x-2)^2 \le 4$  (25.8)

Trong bài toán này, tập xác định D = R nhưng tập khả thi là 0 ≤ x ≤ 4. Đồ thị của hàm mục tiêu được minh hoạ bởi đường nét đậm trong Hình [25.1a.](#page-3-0) Hàm số ràng buộc f1(x) = (x − 2)<sup>2</sup> − 4 được biểu diễn bởi đường chấm gạch. Có thể nhận ra rằng giá trị tối ưu của bài toán là điểm trên đồ thị có hoành độ bằng 0 (là điểm nhỏ nhất trên đường nét đậm trong đoạn [0, 4]). Chú ý rằng hàm mục tiêu không phải là hàm lồi nên bài toán tối ưu cũng không phải là lồi, mặc dù hàm bất phương trình ràng buộc f1(x) là lồi.

Hàm số Lagrange của bài toàn này có dạng

$$\mathcal{L}(x,\lambda) = x^2 + 10\sin(x) + 10 + \lambda((x-2)^2 - 4)$$

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

![](_page_3_Figure_2.jpeg)

Hình 25.1. Ví dụ về hàm số đối ngẫu. (a) Đường nét liền đậm thể hiện hàm mục tiêu. Đường chấm gạch thể hiện hàm số ràng buộc. Các đường chấm chấm thể hiện hàm Lagrange ứng với λ khác nhau. (b) Đường nét đứt nằm ngang thể hiện giá trị tối ưu của bài toán. Đường nét liền thể hiện hàm số đối ngẫu. Với mọi λ, giá trị của hàm đối ngẫu nhỏ hơn hoặc bằng giá trị tối ưu của bài toán chính.

Các đường nét chấm trong Hình [25.1a](#page-3-0) là các đồ thị của hàm Lagrange ứng với λ khác nhau. Vùng bị chặn giữa hai đường thẳng đứng màu đen thể hiện tập khả thi của bài toán.

Với mỗi λ, hàm số đối ngẫu được định nghĩa là:

$$g(\lambda) = \inf_{x} (x^2 + 10\sin(x) + 10 + \lambda((x-2)^2 - 4)), \quad \lambda \ge 0.$$

Từ Hình [25.1a,](#page-3-0) có thể thấy rằng với các λ khác nhau, hàm g(λ) đạt giá trị nhỏ nhất tại điểm có hoành độ bằng 0 của đường nét liền hoặc tại một điểm thấp hơn điểm đó. Trong Hình [25.1b,](#page-3-0) đường nét liền thể hiện đồ thị của hàm g(λ), đường nét đứt thể hiện giá trị tối ưu của bài toán tối ưu chính. Ta có thể thấy hai điều:

- Đường nét liền luôn nằm phía dưới (hoặc có đoạn trùng) đường nét đứt.
- Hàm g(λ) là một hàm lõm.

Mã nguồn cho Hình [25.1](#page-3-0) có thể được tìm thấy tại <https://goo.gl/jZiRCp>.

Ví dụ 2 : Xét một bài toán quy hoạch tuyến tính:

$$x = \arg\min_{\mathbf{x}} \mathbf{c}^{T} \mathbf{x}$$
thoả mãn:  $\mathbf{A}\mathbf{x} = \mathbf{b}$  (25.9) 
$$\mathbf{x} \succeq 0$$

Hàm ràng buộc cuối cùng có thể được viết lại thành  $f_i(\mathbf{x}) = -x_i, i = 1, \dots, n$ . Hàm Lagrange của bài toán này là:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \mathbf{c}^T \mathbf{x} - \sum_{i=1}^n \lambda_i x_i + \boldsymbol{\nu}^T (\mathbf{A} \mathbf{x} - \mathbf{b}) = -\mathbf{b}^T \boldsymbol{\nu} + (\mathbf{c} + \mathbf{A}^T \boldsymbol{\nu} - \boldsymbol{\lambda})^T \mathbf{x}$$

(đừng quên điều kiện  $\lambda \succeq 0$ ). Hàm đối ngẫu là

$$g(\lambda, \nu) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda, \nu) = -\mathbf{b}^T \nu + \inf_{\mathbf{x}} (\mathbf{c} + \mathbf{A}^T \nu - \lambda)^T \mathbf{x}$$
 (25.10)

Nhận thấy rằng hàm tuyến tính  $\mathbf{d}^T\mathbf{x}$  của  $\mathbf{x}$  bị chặn dưới khi vào chỉ khi  $\mathbf{d}=0$ . Vì nếu có một phần tử  $d_i$  của  $\mathbf{d}$  khác 0, chỉ cần chọn  $x_i$  rất lớn và ngược dấu với  $d_i$ , ta sẽ có một giá trị nhỏ tuỳ ý. Nói cách khác,  $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = -\infty$  trừ khi  $\mathbf{c} + \mathbf{A}^T \boldsymbol{\nu} - \boldsymbol{\lambda} = 0$ . Tóm lại,

$$g(\lambda, \nu) = \begin{cases} -\mathbf{b}^T \nu & \text{n\'eu } \mathbf{c} + \mathbf{A}^T \nu - \lambda = 0 \\ -\infty & \text{o.w.} \end{cases}$$
 (25.11)

Trường hợp thứ hai khi  $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = -\infty$  chúng ta sẽ gặp rất nhiều sau này. Trường hợp này không mấy thú vị vì hiển nhiên  $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq p^*$ . Với mục đích chính là đi tìm chặn dưới của  $p^*$ , ta chỉ cần quan tâm tới các giá trị của  $\boldsymbol{\lambda}$  và  $\boldsymbol{\nu}$  sao cho  $g(\boldsymbol{\lambda}, \boldsymbol{\nu})$  càng lớn càng tốt. Trong bài toán này, ta sẽ quan tâm tới  $\boldsymbol{\lambda}$  và  $\boldsymbol{\nu}$  sao cho  $\mathbf{c} + \mathbf{A}^T \boldsymbol{\nu} - \boldsymbol{\lambda} = 0$ .

# 25.3. Bài toán đối ngẫu Lagrange

Với mỗi cặp  $(\lambda, \nu)$ , hàm đối ngẫu Lagrange cho chúng ta một chặn dưới cho giá trị tối ưu  $p^*$  của bài toán chính (25.5). Câu hỏi đặt ra là: với cặp giá trị nào của  $(\lambda, \nu)$ , chúng ta sẽ có một chặn dưới tốt nhất của  $p^*$ ? Nói cách khác, ta đi cần giải bài toán

<span id="page-4-0"></span>
$$\pmb{\lambda}^*, \pmb{\nu}^* = \arg\max_{\pmb{\lambda},\pmb{\nu}} g(\pmb{\lambda},\pmb{\nu})$$
thoả mãn:  $\pmb{\lambda} \succeq 0$  (25.12)

Đây là một bài toán tối ưu lồi vì ta cần tối đa một hàm lõm trên tập khả thi lồi. Trong nhiều trường hợp, lời giải cho bài toán (25.12) có thể dễ tìm hơn bài toán chính.

Bài toán tối ưu (25.12) được gọi là bài toán đối ngẫu Lagrange (Lagrange dual problem) (hoặc viết gọn là bài toán đối ngẫu) ứng với bài toán chính (25.5). Tập khả thi của bài toán đối ngẫu được gọi là tập khả thi đối ngẫu (dual feasible set). Ràng buộc của bài toán đối ngẫu bao gồm điều kiện  $\lambda \succeq 0$  và điều kiện ẩn  $g(\lambda, \nu) > -\infty$  (điều kiện này được thêm vào vì ta chỉ quan tâm tới các  $(\lambda, \nu)$  sao cho hàm mục tiêu của bài toán đối ngẫu càng lớn càng tốt). Nghiệm của bài toán đối ngẫu (25.12) ký hiệu bởi  $(\lambda^*, \nu^*)$ , được gọi là điểm tối ưu đối ngẫu (dual optimal point).

Trong nhiều trường hợp, điều kiện ẩn  $g(\lambda, \nu) > -\infty$  cũng có thể được viết cụ thể. Quay lại với ví dụ phía trên, điệu kiện ẩn có thể được viết thành  $\mathbf{c} + \mathbf{A}^T \nu - \lambda = 0$ . Đây là một hàm affine. Vì vậy, khi có thêm ràng buộc này, ta vẫn thu được một bài toán lồi.

#### 25.3.1. Đối ngẫu yếu

Ký hiệu giá trị tối ưu của bài toán đối ngẫu (25.12) là  $d^*$ . Theo (25.7), ta đã biết  $d^* \leq p^*$ . Tính chất quan trọng này được gọi là đối ngẫu yếu (weak duality). Ta quan sát thấy hai điều:

- Nếu giá trị tối ưu trong bài toán chính là  $p^* = -\infty$ , ta phải có  $d^* = -\infty$ . Điều này tương đương với việc bài toán đối ngẫu là bất khả thi (không có giá trị nào thỏa mãn các ràng buộc).
- Nếu hàm mục tiêu trong bài toán đối ngẫu không bị chặn trên, nghĩa là  $d^* = +\infty$ , ta phải có  $p^* = +\infty$ . Khi đó, bài toán chính là bất khả thi.

Giá trị  $p^* - d^*$  được gọi là *cách biệt đối ngẫu tối ưu* (optimal duality gap). Cách biệt này luôn là một số không âm.

Đôi khi có những bài toán tối ưu (lồi hoặc không) rất khó giải. Tuy nhiên, nếu tìm được  $d^*$ , ta có thể biết chặn dưới của bài toán chính. Việc tìm  $d^*$  thường đơn giản hơn vì bài toán đối ngẫu luôn luôn là lồi.

## 25.3.2. Đối ngẫu mạnh và tiêu chuẩn ràng buộc Slater

Nếu đẳng thức  $p^* = d^*$  thoả mãn, cách biệt đối ngẫu tối ưu bằng không, ta nói rằng đối ngẫu mạnh (strong duality) xảy ra. Lúc này, việc giải bài toán đối ngẫu đã giúp tìm được chính xác giá trị tối ưu của bài toán gốc.

Thật không may, đối ngẫu mạnh không thường xuyên xảy ra trong các bài toán tối ưu. Tuy nhiên, nếu bài toán chính là lồi, tức có dạng

<span id="page-5-0"></span>
$$x = \arg\min_{\mathbf{x}} f_0(\mathbf{x})$$
thoả mãn:  $f_i(\mathbf{x}) \le 0, i = 1, 2, \dots, m$  (25.13)  
$$\mathbf{A}\mathbf{x} = \mathbf{b}$$

trong đó  $f_0, f_1, \ldots, f_m$  là các hàm lồi, chúng ta thường (không luôn luôn) có đối ngẫu mạnh. Rất nhiều nghiên cứu đã thiết lập các điều kiện ngoài tính chất lồi để đối ngẫu mạnh xảy ra. Những điều kiện đó có tên là  $ti\hat{e}u$  chuẩn ràng  $bu\hat{o}c$  (constraint qualification).

Một trong các tiêu chuẩn ràng buộc phổ biến nhất là *tiêu chuẩn ràng buộc Slater* (Slater's constraint qualification).

Trước khi thảo luận về tiêu chuẩn ràng buộc Slatter, chúng ta cần định nghĩa:

#### Định nghĩa 25.1: Khả thi chặt

Một điểm khả thi của bài toán [\(25.13\)](#page-5-0) được gọi là khả thi chặt (stricly feasible) nếu:

$$f_i(\mathbf{x}) < 0, \ i = 1, 2, \dots, m, \quad \mathbf{A}\mathbf{x} = \mathbf{b}$$

Khả thi chặt khác với khả thi ở việc dấu bằng trong các bất phương trình ràng buộc không xảy ra.

#### Định lý 25.1: Tiêu chuẩn ràng buộc Slater

Nếu bài toàn chính là một bài toán tối ưu lồi và tồn tại một điểm khả thi chặt thì đối ngẫu mạnh xảy ra.

Điều kiện khá đơn giản sẽ giúp ích cho nhiều bài toán tối ưu sau này.

Chú ý:

- Đối ngẫu mạnh không thường xuyên xảy ra. Với các bài toán lồi, điều này xảy ra thường xuyên hơn. Tồn tại những bài toán lồi mà đối ngẫu mạnh không đạt được.
- Có những bài toán không lồi nhưngđối ngẫu mạnh vẫn xảy ra. Bài toán tối ưu trong Hình [25.1](#page-3-0) là một ví dụ.

# 25.4. Các điều kiện tối ưu

#### 25.4.1. Sự lỏng lẻo bù trừ

Giả sử đối ngẫu mạnh xảy ra. Gọi x ∗ là một điểm tối ưu của bài toán chính và (λ ∗ , ν ∗ ) là cặp điểm tối ưu của bài toán đối ngẫu. Ta có

<span id="page-6-0"></span>
$$f_0(\mathbf{x}^*) = g(\boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) \tag{25.14}$$

$$= \inf_{\mathbf{x}} \left( f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i^* f_i(\mathbf{x}) + \sum_{j=1}^p \nu_j^* h_j(\mathbf{x}) \right)$$
 (25.15)

$$\leq f_0(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* f_i(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* h_j(\mathbf{x}^*)$$
 (25.16)

$$\leq f_0(\mathbf{x}^*) \tag{25.17}$$

Đẳng thức [\(25.14\)](#page-6-0) xảy ra do đối ngẫu mạnh. Đẳng thức [\(25.15\)](#page-6-0) xảy ra do định nghĩa của hàm đối ngẫu. Bất đẳng thức [\(25.16\)](#page-6-0) là hiển nhiên vì infimum của một hàm nhỏ hơn giá trị của hàm đó tại bất kỳ điểm nào khác. Bất đẳng thức [\(25.17\)](#page-6-0) xảy ra vì các ràng buộc fi(x ∗ ) ≤ 0, λ<sup>i</sup> ≥ 0, i = 1, 2, . . . , m và h<sup>j</sup> (x ∗ ) = 0. Từ đây có thể thấy rằng dấu đẳng thức ở [\(25.16\)](#page-6-0) và [\(25.17\)](#page-6-0) phải đồng thời xảy ra. Ta lại có thêm hai quan sát thú vị nữa:

- x ∗ là một điểm tối ưu của g(λ ∗ , ν ∗ ).
- Thú vị hơn, <sup>X</sup><sup>m</sup> i=1 λ ∗ i fi(x ∗ ) = 0. Vì λ ∗ fi(x ∗ ) ≤ 0, ta phải có λ ∗ i fi(x ∗ ) = 0, ∀i.

Điều kiện này được gọi là điều kiện lỏng lẻo bù trừ (complementary slackness). Từ đây ta có:

$$\lambda_i^* > 0 \Rightarrow f_i(\mathbf{x}^*) = 0 \tag{25.18}$$

$$f_i(\mathbf{x}^*) < 0 \Rightarrow \lambda_i^* = 0 \tag{25.19}$$

Tức một trong hai giá trị này phải bằng 0.

#### 25.4.2. Các điều kiện tối ưu KKT

Ta vẫn giả sử rằng các hàm đang xét có đạo hàm và bài toán tối ưu không nhất thiết là lồi.

## Điều kiện KKT cho bài toán tối ưu (không nhất thiết lồi)

Giả sử đối ngẫu mạnh xảy ra. Gọi x <sup>∗</sup> và (λ ∗ , ν ∗ ) là mộ bộ điểm tối ưu chính và tối ưu đối ngẫu. Vì x ∗ tối ưu hàm khả vi L(x,λ ∗ , ν ∗ ), đạo hàm của hàm Lagrange tại x <sup>∗</sup> phải bằng 0.

Điều kiện Karush-Kuhn-Tucker (KKT) nói rằng x ∗ ,λ ∗ , ν <sup>∗</sup> phải thoả mãn:

$$f_i(\mathbf{x}^*) \le 0, i = 1, 2, \dots, m \ (25.20)$$

$$h_j(\mathbf{x}^*) = 0, j = 1, 2, \dots, p \ (25.21)$$

$$\lambda_i^* \ge 0, i = 1, 2, \dots, m (25.22)$$

$$\lambda_i^* f_i(\mathbf{x}^*) = 0, i = 1, 2, \dots, m$$
 (25.23)

$$\nabla_{\mathbf{x}} f_0(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla_{\mathbf{x}} f_i(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* \nabla_{\mathbf{x}} h_j(\mathbf{x}^*) = 0$$
 (25.24)

Đây là điều kiện cần để x ∗ ,λ ∗ , ν ∗ là nghiệm của bài toán chính và bài toán đối ngẫu. Hai điều kiện đầu chính là ràng buộc của bài toán chính. Điều kiện λ ∗ i fi(x ∗ ) là điều kiện lỏng lẻo bù trừ. Điều kiện cuối cùng là đạo hàm của hàm Lagrange theo x <sup>∗</sup> bằng không.

# Điều kiện KKT cho bài toán lồi

Với bài toán lồi và đối ngẫu mạnh xảy ra, các điều kiện KKT vừa đề cập cũng là điều kiện đủ. Vậy với các bài toán tối ưu lồi có hàm mục tiêu và hàm ràng buộc là khả vi, bất kỳ bộ  $(\mathbf{x}*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$  nào thoả mãn các điều kiện KKT đều là điểm tối ưu của bài toán chính và bài toán đối ngẫu.

Các điều kiện KKT rất quan trọng trong tối ưu. Trong một vài trường hợp đặc biệt (chúng ta sẽ thấy trong Phần 26), việc giải hệ (bất) phương trình các điều kiện KKT là khả thi. Rất nhiều thuật toán tối ưu được xây dựng dựa trên việc giải hệ điều kiện KKT.

Ví dụ: Xét bài toán quy hoạch toàn phương với ràng buộc phương trình:

$$\mathbf{x} = \arg\min_{\mathbf{x}} \frac{1}{2} \mathbf{x}^T \mathbf{P} \mathbf{x} + \mathbf{q}^T \mathbf{x} + r$$
thoả mãn:  $\mathbf{A} \mathbf{x} = \mathbf{b}$ . (25.25)

Trong đó  ${\bf P}$ làm một ma trận nửa nửa xác định dương. Hàm số Lagrange của bài toán này là

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\nu}) = \frac{1}{2}\mathbf{x}^T\mathbf{P}\mathbf{x} + \mathbf{q}^T\mathbf{x} + r + \boldsymbol{\nu}^T(\mathbf{A}\mathbf{x} - \mathbf{b})$$

Hệ điều kiện KKT:

$$\mathbf{A}\mathbf{x}^* = \mathbf{b} \tag{25.26}$$

$$\mathbf{P}\mathbf{x}^* + \mathbf{q} + \mathbf{A}^T \boldsymbol{\nu}^* = 0 \tag{25.27}$$

Phương trình thứ hai chính là phương trình đạo hàm của hàm Lagrange tại  $\mathbf{x}^*$  bằng 0. Hệ phương trình này có thể được viết lại dưới dạng

$$\begin{bmatrix} \mathbf{P} & \mathbf{A}^T \\ \mathbf{A} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{x}^* \\ \boldsymbol{\nu}^* \end{bmatrix} = \begin{bmatrix} -\mathbf{q} \\ \mathbf{b} \end{bmatrix}$$

Đây là một phương trình tuyến tính đơn giản.

## 25.5. Tóm tắt

 $Gi \mathring{a} s \mathring{u} r \mathring{a} ng \ c \acute{a} c \ h \grave{a} m \ s \acute{o} \ d \mathring{e} u \ k h \mathring{a} \ v i.$ 

- Các bài toán tối ưu với ràng buộc chỉ gồm phương trình có thể được giải bằng phương pháp nhân tử Lagrange. Điều kiện cần để một điểm là nghiệm của bài toán tối ưu là nó phải thỏa mãn đạo hàm của hàm Lagrange bằng không.
- Với các bài toán tối ưu (không nhất thiết lồi) có thêm ràng buộc là bất phương trình, chúng ta có hàm Lagrange tổng quát và các biến đối ngẫu Lagrange  $\lambda, \nu$ . Với các giá trị  $(\lambda, \nu)$  cố định, ta có định nghĩa về hàm đối ngẫu Lagrange  $g(\lambda, \nu)$ . Hàm số này là infimum của hàm Lagrange khi  $\mathbf{x}$  thay đổi trên tập xác định của bài toán.

- g(λ, ν) ≤ p <sup>∗</sup> với mọi (λ, ν).
- Hàm đối ngẫu Lagrange là lõm bất kể bài toán tối ưu chính có lồi hay không.
- Bài toán đi tìm giá trị lớn nhất của hàm đối ngẫu Lagrange với điều kiện λ 0 được gọi là bài toán đối ngẫu. Đây là một bài toán tối ưu lồi bất kể bài toán chính có lồi hay không.
- Gọi giá trị tối ưu của bài toán đối ngẫu là d ∗ , ta có d <sup>∗</sup> ≤ p ∗ . Đây được gọi là đối ngẫu yếu.
- Đối ngẫu mạnh xảy ra khi d <sup>∗</sup> = p ∗ . Trong các bài toán lồi, đối ngẫu mạnh thường xảy ra nhiều hơn.
- Nếu bài toán chính là lồi và tiêu chuẩn ràng buộc Slater thoả mãn thì đối ngẫu mạnh xảy ra.
- Nếu bài toán chính là lồi và đối ngẫu mạnh xảy ra thì điểm tối ưu thoả mãn các điều kiện KKT (điều kiện cần và đủ).
- Rất nhiều bài toán tối ưu được giải quyết thông qua các điều kiện KKT.