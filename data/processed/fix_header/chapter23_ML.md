# Phần VII

Tối ưu lồi

Tập lồi và hàm lồi

## 23.1. Giới thiệu

### 23.1.1. Bài toán tối ưu

Các bài toán tối ưu đã thảo luận trong cuốn sách này đều là các bài toán tối ưu không ràng buộc (unconstrained optimization problem), tức tối ưu hàm mất mát mà không có điều kiện ràng buộc (constraint) nào về nghiệm. Tuy nhiên, không chỉ trong machine learning, cài bài toán tối ưu trên thực tế thường có rất nhiều ràng buộc khác nhau.

Trong toán tối ưu, một bài toán có ràng buộc thường được viết dưới dạng

<span id="page-1-1"></span>
$$\mathbf{x}^* = \arg\min_{\mathbf{x}} f_0(\mathbf{x})$$
thỏa mãn: 
$$f_i(\mathbf{x}) \leq 0, \quad i = 1, 2, \dots, m$$
$$h_j(\mathbf{x}) = 0, \quad j = 1, 2, \dots, p$$
 (23.1)

Trong đó, vector x = [x1, x2, . . . , xn] <sup>T</sup> được gọi là biến tối ưu (optimization variable). Hàm số f<sup>0</sup> : R <sup>n</sup> → R được gọi là hàm mục tiêu (objective function)[60](#page-1-0). Các bất phương trình fi(x) <= 0, i = 1, 2, . . . , m được gọi là bất phương trình ràng buộc (inequality constraint), và các hàm tương ứng fi(x), i = 1, 2, . . . , m được gọi là hàm bất phương trình ràng buộc (inequality constraint function). Các phương trình h<sup>j</sup> (x) = 0, j = 1, 2, . . . , p được gọi là các phương trình ràng buộc (equality constraint), các hàm tương ứng là các hàm phương trình ràng buộc (equality constraint function).

<span id="page-1-0"></span><sup>60</sup> các hàm mục tiêu trong machine learning thường được gọi là hàm mất mát

Ký hiệu  $\mathbf{dom} f$  là tập các điểm mà trên đó hàm số xác định, hay còn gọi là tập xác định (domain). Tập xác định của một bài toán tối ưu là giao của tập xác định tất cả các hàm liên quan:

$$\mathcal{D} = \bigcap_{i=0}^{m} \mathbf{dom} f_i \cap \bigcap_{j=0}^{p} \mathbf{dom} h_j$$
 (23.2)

Một điểm  $\mathbf{x} \in \mathcal{D}$  được gọi là  $diểm\ khẩ\ thi$  (feasible point) nếu nó thỏa mãn tất cả ràng buộc:  $f_i(\mathbf{x}) \leq 0, i = 1, 2, \dots, m; h_j(\mathbf{x}) = 0, j = 1, 2, \dots, p$ . Tập hợp các điểm khẩ thi được gọi là  $tập\ khẩ\ thi$  (feasible set) hoặc  $tập\ ràng\ buộc$  (constraint set). Như vậy, tập khẩ thi là một tập con của tập xác định. Mỗi điểm trong tập khẩ thi được gọi là một  $diểm\ khẩ\ thi$  (feasible point).

Bài toán (23.1) được gọi là khå thi (tương ứng bắt khå thi) nếu tập khả thi của nó khác (tương ứng bằng) rỗng.

Chú ý:

- Nếu bài toán yêu cầu tìm giá trị lớn nhất thay vì nhỏ nhất của hàm mục tiêu, ta chỉ cần đổi dấu của  $f_0(\mathbf{x})$ .
- Nếu ràng buộc là lớn hơn hoặc bằng  $(\geq)$ , tức  $f_i(\mathbf{x}) \geq b_i$ , ta chỉ cần đổi dấu của ràng buộc là sẽ có điều kiện nhỏ hơn hoặc bằng  $-f_i(\mathbf{x}) \leq -b_i$ .
- $\bullet$  Các ràng buộc cũng có thể là lớn hơn (>) hoặc nhỏ hơn (<).
- Nếu ràng buộc là bằng nhau, tức  $h_j(\mathbf{x}) = 0$ , ta có thể viết nó dưới dạng hai bất phương trình  $h_j(\mathbf{x}) \leq 0$  và  $-h_j(\mathbf{x}) \leq 0$ .
- Trong chương này,  $\mathbf{x}$ ,  $\mathbf{y}$  được dùng chủ yếu để ký hiệu các biến số, không phải là dữ liệu như trong các chương trước. Các biến cần tối được ghi dưới dấu arg min. Khi viết một bài toán tối ưu, ta cần chỉ rõ biến nào cần được tối ưu, biến nào là cố định.

Nhìn chung, không có cách giải quyết tổng quát cho các bài toán tối ưu, thậm chí nhiều bài toán tối ưu chưa có lời giải hiểu quả. Hầu hết các phương pháp không chứng minh được nghiệm tìm được có phải là điểm tối ưu toàn cục hay không. Thay vào đó, nghiệm thường là các điểm cực trị địa phương. Trong nhiều trường hợp, các cực trị địa phương cũng mang lại những kết quả tốt.

Để bắt đầu nghiên cứu về tối ưu, chúng ta cần biết tới một mảng rất quan trọng có tên là tối ưu lồi (convex optimization), trong đó hàm mục tiêu là một hàm lồi (convex function), tập khả thi là một tập lồi (convex set). Những tính chất đặc biệt về cực trị địa phương và toàn cục của một hàm lồi khiến tối ưu lồi trở nên cực kỳ quan trọng. Trong chương này, chúng ta sẽ thảo luận định nghĩa và các

<span id="page-3-0"></span>![](_page_3_Picture_1.jpeg)

Hình 23.1. Các ví dụ về tập lồi

tính chất cơ bản của tập lồi và hàm lồi.  $Bài\ toán\ tối\ uu\ lồi\ (convex\ optimization\ problem)$  sẽ được đề cập trong chương tiếp theo.

Trước khi đi sâu vào tập lồi và hàm lồi, xin nhắc lại các hàm liên quan: supremum và infimum.

### 23.1.2. Các hàm supremum và infimum

Xét một tập  $\mathcal{C} \subset \mathbb{R}$ . Một số a được gọi là chặn trên (upper bound) của  $\mathcal{C}$  nếu  $x \leq a, \ \forall x \in \mathcal{C}$ . Tập các chặn trên của một tập hợp có thể là tập rỗng, ví dụ  $\mathcal{C} \equiv \mathbb{R}$ , toàn bộ  $\mathbb{R}$ , chỉ khi  $\mathcal{C} =$ ), hoặc nửa đoạn  $[b, +\infty)$ . Trong trường hợp cuối, số b được gọi là chặn trên nhỏ nhất (supremum) của  $\mathcal{C}$ , được ký hiệu là sup  $\mathcal{C}$ . Chúng ta cũng ký hiệu sup  $= -\infty$  và sup  $\mathcal{C} = +\infty$  nếu  $\mathcal{C}$  không bi chặn trên (unbounded above).

Tương tự, một số a được gọi là chặn dưới (lower bound) của  $\mathcal{C}$  nếu  $x \geq a$ ,  $\forall x \in \mathcal{C}$ . Chặn dưới lớn <math>nhất (infimum) của  $\mathcal{C}$  được ký hiệu là inf  $\mathcal{C}$ . Chúng ta cũng định nghĩa inf  $= +\infty$  và inf  $\mathcal{C} = -\infty$  nếu  $\mathcal{C}$  không bị chặn dưới.

Nếu  $\mathcal C$  có hữu hạn số phần tử thì  $\max \mathcal C = \sup \mathcal C$  và  $\min \mathcal C = \inf \mathcal C$ .

## 23.2. Tập lồi

### 23.2.1. Định nghĩa

Bạn đoc có thể đã biết đến khái niệm da giác lồi. Hiểu một cách đơn giản, lồi (convex) là phình ra ngoài, hoặc nhô ra ngoài. Trong toán học, bằng phẳng cũng được coi là lồi.

 $\bm{Dinh}$  nghĩa không chính thức của tập lồi: Một tập hợp được gọi là tập lồi (convex set) nếu mọi điểm trên đoạn thẳng nối hai điểm bất kỳ trong nó đều thuộc tập hợp đó.

Một vài ví dụ về tập lồi được cho trong Hình 23.1. Các hình với đường biên màu đen thể hiện việc biên cũng thuộc vào hình đó, biên màu trắng thể hiện việc biên đó không nằm trong hình. Đường thẳng hoặc đoạn thẳng cũng là một tập lồi theo định nghĩa phía trên.

<span id="page-4-0"></span>![](_page_4_Picture_1.jpeg)

Hình 23.2. Các ví dụ về tập không lồi.

Một vài ví dụ thực tế:

- Giả sử một căn phòng có dạng hình lồi, nếu ta đặt một bóng đèn đủ sáng ở bất kỳ vị trí nào trên trần nhà, mọi điểm trong căn phòng đều được chiếu sáng.
- Nếu một đất nước có bản đồ dạng hình lồi thì đoạn thẳng nối hai thành phố bất kỳ của đất nước đó nằm trọn vẹn trong lãnh thổ của nó. Một cách lý tưởng, mọi đường bay trong đất nước đều được tối ưu vì chi phí bay thẳng ít hơn chi phí bay vòng hoặc qua không phận của nước khác. Bản đồ Việt Nam không có dạng lồi vì đường thẳng nối sân bay Nội Bài và Tân Sơn Nhất đi qua địa phận Campuchia.

Hình [23.2](#page-4-0) minh hoạ một vài ví dụ về các tập không phải là tập lồi, nói gọn là tập không lồi (nonconvex set). Ba hình đầu tiên không phải lồi vì các đường nét đứt chứa nhiều điểm không nằm bên trong các tập đó. Hình thứ tư, hình vuông không có biên ở đáy, là tập không lồi vì đoạn thẳng nối hai điểm ở đáy có thể chứa phần ở giữa không thuộc tập đang xét. Một đường cong bất kỳ cũng là tập không lồi vì đường thẳng nối hai điểm bất kỳ không thuộc đường cong đó.

Để mô tả một tập lồi dưới dạng toán học, ta sử dụng:

#### <span id="page-4-1"></span>Định nghĩa 23.1: Tập hợp lồi

Một tập hợp C được gọi là một tập lồi (convex set) nếu với hai điểm bất kỳ x1, x<sup>2</sup> ∈ C, điểm x<sup>θ</sup> = θx<sup>1</sup> + (1 − θ)x<sup>2</sup> cũng nằm trong C với 0 ≤ θ ≤ 1.

Tập hợp các điểm có dạng (θx<sup>1</sup> + (1 − θ)x2) chính là đoạn thẳng nối hai điểm x<sup>1</sup> và x2.

Với định nghĩa này, toàn bộ không gian là một tập lồi vì đoạn thằng nào cũng nằm trong không gian đó. Tập rỗng cũng có thể coi là một trường hợp đặc biệt của tập lồi.

### 23.2.2. Các ví dụ về tập lồi

Siêu mặt phẳng và nửa không gian

#### Định nghĩa 23.2: Siêu mặt phẳng

Một  $si\hat{e}u$  mặt phẳng, hay  $si\hat{e}u$  phẳng (hyperplane) trong không gian n chiều là tập hợp các điểm thỏa mãn phương trình

$$a_1x_1 + a_2x_2 + \dots + a_nx_n = \mathbf{a}^T\mathbf{x} = b$$
 (23.3)

với  $b, a_i, i = 1, 2, ..., n$  là các số thực.

Siêu phẳng là các tập lồi. Điều này có thể được suy ra từ Định nghĩa 23.1. Thật vây, nếu

$$\mathbf{a}^T \mathbf{x}_1 = \mathbf{a}^T \mathbf{x}_2 = b$$

thì với  $0 \le \theta \le 1$  bất kỳ, ta có  $\mathbf{a}^T \mathbf{x}_{\theta} = \mathbf{a}^T (\theta \mathbf{x}_1 + (1 - \theta) \mathbf{x}_2) = \theta b + (1 - \theta) b = b$ . Tức là  $\theta \mathbf{x}_1 + (1 - \theta) \mathbf{x}_2$  cũng là một điểm thuộc siêu phẳng đó.

#### Định nghĩa 23.3: Nửa không gian

Một  $n \dot{t} a \ không gian$  (halfspace) trong không gian n chiều là tập hợp các điểm thỏa mãn bất phương trình

$$a_1x_1 + a_2x_2 + \dots + a_nx_n = \mathbf{a}^T\mathbf{x} \le b$$

với  $b, a_i, i = 1, 2, \dots, n$  là các số thực.

Các nửa không gian cũng là các tập lồi, bạn đọc có thể kiểm tra theo Định nghĩa 23.1 và cách chứng minh tương tự như trên.

Cầu chuẩn

#### Định nghĩa 23.4: Cầu chuẩn

Cho một tâm  $\mathbf{x}_c$ , một bán kính r và khoảng cách giữa các điểm được xác định bởi một chuẩn. Cầu chuẩn (norm ball) tương ứng là tập hợp các điểm thoả mãn

$$B(\mathbf{x}_c, r) = \{\mathbf{x} \mid ||\mathbf{x} - \mathbf{x}_c|| \le r\} = \{\mathbf{x}_c + r\mathbf{u} \mid ||\mathbf{u}|| \le 1\}$$

Khi chuẩn là  $\ell_2$ , cầu chuẩn là một hình tròn trong không gian hai chiều, hình cầu trong không gian ba chiều, hoặc siêu cầu trong các không gian nhiều chiều.

Cầu chuẩn là tập lồi. Để chứng minh việc này, ta dùng Định nghĩa 23.1 và bất đẳng thức tam giác của chuẩn. Với  $\mathbf{x}_1, \mathbf{x}_2$  bất kỳ thuộc  $B(\mathbf{x}_c, r)$  và  $0 \le \theta \le 1$  bất kỳ, xét  $\mathbf{x}_{\theta} = \theta \mathbf{x}_1 + (1 - \theta) \mathbf{x}_2$ , ta có:

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

Hình 23.3. Hình dạng của các tập hợp bị chặn bởi các (a) giả chuẩn và (b) chuẩn.

$$\|\mathbf{x}_{\theta} - \mathbf{x}_{c}\| = \|\theta(\mathbf{x}_{1} - \mathbf{x}_{c}) + (1 - \theta)(\mathbf{x}_{2} - \mathbf{x}_{c})\|$$

$$\leq \theta \|\mathbf{x}_{1} - \mathbf{x}_{c}\| + (1 - \theta)\|\mathbf{x}_{2} - \mathbf{x}_{c}\| \leq \theta r + (1 - \theta)r = r$$

Vậy  $\mathbf{x}_{\theta} \in B(\mathbf{x}_{c}, r)$ .

Hình 23.3 minh họa tập hợp các điểm có tọa độ (x,y) trong không gian hai chiều thỏa mãn:

<span id="page-6-1"></span>
$$(|x|^p + |y|^p)^{1/p} \le 1 (23.4)$$

Hàng trên là các tập tương ứng  $0 là các giả chuẩn; hàng dưới tương ứng <math>p \geq 1$  là các chuẩn thực sự. Có thể thấy rằng khi p nhỏ gần bằng không, tập hợp các điểm thỏa mãn bất đẳng thức (23.4) gần như nằm trên các trực tọa độ và bị chặn trong đoạn [0,1]. Quan sát này sẽ giúp ích khi làm việc với giả chuẩn  $\ell_0$ . Khi  $p \to \infty$ , các tập hợp hội tự về hình vuông. Đây cũng là một trong các lý do vì sao cần có điều kiện  $p \geq 1$  khi định nghĩa chuẩn  $\ell_p$ .

### 23.2.3. Giao của các tập lồi

Giao của các tập lồi là một tập lồi. Điều này có thể nhận thấy trong Hình 23.4a. Giao của hai trong ba hoặc cả ba tập lồi đều là các tập lồi. Điều này có thể được chứng minh theo Định nghĩa 23.1: nếu  $\mathbf{x}_1, \mathbf{x}_2$  thuộc giao của các tập lồi thì  $(\theta \mathbf{x}_1 + (1 - \theta) \mathbf{x}_2)$  cũng thuộc giao của chúng.

Từ đó suy ra giao của các nửa không gian và nửa mặt phẳng là một tập lồi. Chúng là các đa giác lồi trong không gian hai chiều, đa diện lồi trong không gian ba chiều,

<span id="page-7-0"></span>![](_page_7_Picture_1.jpeg)

**Hình 23.4.** (a) Giao của các tập lồi là một tập lồi. (b) Giao của các siêu phẳng và nửa không gian là một tập lồi và được gọi là *siêu đa diện* (polyhedra).

và  $si\hat{e}u$  đa  $di\hat{e}n$  trong không gian nhiều chiều. Giả sử có m nửa không gian và p siêu phẳng. Mỗi nửa không gian có thể được viết dưới dạng  $\mathbf{a}_i^T\mathbf{x} \leq b_i, \ \forall i=1,2,\ldots,m$ . Một siêu phẳng có thể được viết dưới dạng  $\mathbf{c}_i^T\mathbf{x} = d_i, \ \forall i=1,2,\ldots,p$ .

Vậy nếu đặt  $\mathbf{A} = [\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_m], \mathbf{b} = [b_1, b_2, \dots, b_m]^T, \mathbf{C} = [\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_p]$  và  $\mathbf{d} = [d_1, d_2, \dots, d_p]^T$ , ta có thể viết siêu đa diện dưới dạng tập hợp các điểm  $\mathbf{x}$  thỏa mãn

$$\mathbf{A}^T \mathbf{x} \leq \mathbf{b}, \quad \mathbf{C}^T \mathbf{x} = \mathbf{d}$$

trong đó  $\preceq$  thể hiện mỗi phần tử trong vế trái nhỏ hơn hoặc bằng phần tử tương ứng trong vế phải.

### 23.2.4. Tổ hợp lồi và bao lồi

#### Định nghĩa 23.5: Tổ hợp lồi

Một điểm được gọi là  $t\hat{o}$  hợp  $l\hat{o}i$  (convex combination) của các điểm  $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_k$  nếu nó có thể được viết dưới dạng

$$\mathbf{x} = \theta_1 \mathbf{x}_1 + \theta_2 \mathbf{x}_2 + \dots + \theta_k \mathbf{x}_k \ \text{v\'oi} \ \theta_1 + \theta_2 + \dots + \theta_k = 1 \ \text{v\'a} \ \theta_i \geq 0, \forall i=1,2,\dots,k$$

 $Bao\ lồi\ (convex\ hull)$  của một tập bất kỳ là tập toàn bộ các tổ hợp lồi của tập hợp đó. Bao lồi của một tập bất kỳ là một tập lồi. Bao lồi của một tập lồi là chính nó. Bao lồi của một tập hợp là tập lồi nhỏ nhất chứa tập hợp đó. Khái niệm  $nhổ\ nhất\ dược\ hiểu\ là\ mọi\ tập lồi\ chứa toàn bộ các tổ hợp lồi đều chứa bao lồi của tập hợp đó.$ 

Nhắc lại khái niệm *tách biệt tuyến tính* đã sử dụng nhiều trong cuốn sách. Hai tập hợp được gọi là tách biệt tuyến tính nếu bao lồi của chúng không giao nhau.

Trong Hình 23.5, bao lồi của các điểm màu đen là vùng màu xám bao bởi các đa giác lồi. Trong Hình 23.5 bên phải, bao lồi của đa giác lõm nền chấm là hợp của nó và phần tam giác màu xám.

<span id="page-8-0"></span>![](_page_8_Picture_1.jpeg)

Hình 23.5. Trái: Bao lồi của các điểm màu đen là đa giác lồi nhỏ nhất chứa toàn bộ các điểm này. Phải: Bao lồi của đa giác lõm nền chấm là hợp của nó và tam giác màu xám phía trên. Hai bao lồi không giao nhau này có thể được phân tách hoàn toàn bằng một siêu phẳng (trong trường hơp này là một đường thẳng).

#### Định lý 23.1: Siêu phẳng phân chia

Hai tập lồi không rỗng C, D không giao nhau khi và chỉ khi tồn tại một vector a và một số b sao cho

$$\mathbf{a}^T\mathbf{x} \leq b, \forall \mathbf{x} \in \mathcal{C}, \quad \mathbf{a}^T\mathbf{x} \geq b, \forall \mathbf{x} \in \mathcal{D}$$

Tập hợp tất cả các điểm x thỏa mãn a <sup>T</sup> x = b là một siêu phẳng. Siêu phẳng này được gọi là siêu phẳng phân chia (separating hyperplane).

Ngoài ra, còn nhiều tính chất thú vị của các tập lồi và các phép toán bảo toàn chính chất lồi của một tập hợp, bạn đọc có thể đọc thêm Chương 2 của cuốn Convex Optimization [BV04].

## 23.3. Hàm lồi

### 23.3.1. Định nghĩa

Trước hết ta xem xét các hàm một biến với đồ thị của nó là một đường trong một mặt phẳng. Một hàm số được gọi là lồi (convex) nếu tập xác định của nó là một tập lồi và đoạn thẳng nối hai điểm bất kỳ trên đồ thị hàm số đó nằm về phía trên hoặc nằm trên đồ thị (xem Hình [23.6\)](#page-9-0).

<span id="page-9-0"></span>![](_page_9_Figure_1.jpeg)

Hình 23.6. Định nghĩa hàm lồi. Diễn đạt bằng lời, một hàm số là lồi nếu đoạn thẳng nối hai điểm bất kỳ trên đồ thị của nó không nằm phía dưới đồ thị đó.

#### <span id="page-9-1"></span>Định nghĩa 23.6: Hàm lồi

Một hàm số f : R <sup>n</sup> → R được gọi là một hàm lồi (convex function) nếu domf là một tập lồi, và:

$$f(\theta \mathbf{x} + (1 - \theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1 - \theta)f(\mathbf{y})$$

với mọi x, y ∈ domf, 0 ≤ θ ≤ 1.

Điều kiện domf là một tập lồi rất quan trọng. Nếu không có điều kiện này, tồn tại những θ mà θx<sup>1</sup> + (1 − θ)x<sup>2</sup> không thuộc domf và f(θx + (1 − θ)y) không xác định.

Một hàm số f được gọi là hàm lõm (concave fucntion) nếu −f là một hàm lồi. Một hàm số có thể không thuộc hai loại trên. Các hàm tuyến tính vừa lồi vừa lõm.

#### Định nghĩa 23.7: Hàm lồi chặt

Một hàm số f : R <sup>n</sup> → R được gọi là hàm lồi chặt (strictly convex function) nếu domf là một tập lồi, và

$$f(\theta \mathbf{x} + (1 - \theta)\mathbf{y}) < \theta f(\mathbf{x}) + (1 - \theta)f(\mathbf{y})$$

∀ x, y ∈ domf, x 6= y, 0 < θ < 1 (chỉ khác hàm lồi ở dấu nhỏ hơn).

Tương tự với định nghĩa hàm lõm chặt (stricly concave function).

Nếu một hàm số là lồi chặt và có điểm cực trị, thì điểm cực trị đó là duy nhất và cũng là cực trị toàn cục.

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

Hình 23.7. Ví dụ về Pointwise maximum. Maximum của các hàm lồi là một hàm lồi.

### 23.3.2. Các tính chất cơ bản

- Nếu f(x) là một hàm lồi thì af(x) cũng lồi khi a > 0 và lõm khi a < 0. Điều này có thể suy ra trực tiếp từ định nghĩa.
- Tổng của hai hàm lồi là một hàm lồi, với tập xác định là giao của hai tập xác định của hai hàm đã cho (nhắc lại rằng giao của hai tập lồi là một tập lồi).
- Hàm max và sup tại từng điểm: Nếu các hàm số f1, f2, . . . , f<sup>m</sup> lồi thì

$$f(\mathbf{x}) = \max\{f_1(\mathbf{x}), f_2(\mathbf{x}), \dots, f_m(\mathbf{x})\}\$$

cũng là lồi trên domf = \m i=1 domf<sup>i</sup> . Hàm max cũng có thể thay thế bằng hàm

sup. Tính chất này có thể được chứng minh theo Định nghĩa [23.6.](#page-9-1) Hình [23.7](#page-10-0) minh hoạ tính chất này. Các hàm f1(x), f2(x) là các hàm lồi. Đường nét đậm chính là đồ thị của hàm số f(x) = max(f1(x), f2(x)). Mọi đoạn thẳng nối hai điểm bất kì trên đường này đều không nằm phía dưới nó.

### 23.3.3. Ví dụ

Các hàm một biến

Ví dụ về hàm lồi:

- Hàm y = ax + b là một hàm lồi vì đoạn thẳng nối hai điểm bất kỳ trên đường thẳng đó đều không nằm phía dưới đường thẳng đó.
- Hàm y = e ax với a ∈ R bất kỳ.
- Hàm y = x a trên tập các số thực dương và a ≥ 1 hoặc a ≤ 0.
- Hàm entropy âm (negative entropy) y = x log x trên tập các số thực dương.

Hình [23.8](#page-11-0) minh hoạ đồ thị của một số hàm lồi thường gặp với biến một chiều.

<span id="page-11-0"></span>![](_page_11_Figure_1.jpeg)

Hình 23.8. Ví dụ về các hàm lồi một biến.

<span id="page-11-1"></span>![](_page_11_Figure_3.jpeg)

Hình 23.9. Ví dụ về các hàm lõm một biến.

Ví dụ về hàm lõm:

- Hàm y = ax + b là một concave function vì -y là một convex function.
- Hàm  $y=x^a$  trên tập số dương và  $0 \le a \le 1$ .
- Hàm logarith<br/>m $y = \log(x)$ trên tập các số dương.

Hình 23.9 minh hoạ đồ thị của một vài hàm số concave.

Hàm affine

Hàm affine là tổng của một hàm tuyến tính và một hằng số, tức là các hàm có dạng  $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x} + b$ .

Khi biến là một ma trận  $\mathbf{X}$ , các hàm affine được định nghĩa:

$$f(\mathbf{X}) = \operatorname{trace}(\mathbf{A}^T \mathbf{X}) + b$$

trong đó,  ${\bf A}$  là một ma trận có cùng kích thước như  ${\bf X}$  để đảm bảo phép nhân ma trận thực hiện được và kết quả là một ma trận vuông. Các hàm affine vừa lồi vừa lõm.

Dạng toàn phương

Da thức bậc hai một biến có dạng  $f(x) = ax^2 + bx + c$  là lồi nếu a > 0, là lõm nếu a < 0.

Khi biến là một vector  $\mathbf{x} = [x_1, x_2, \dots, x_n]$ , dạng toàn phương (quadratic form) là một hàm số có dạng

$$f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x} + \mathbf{b}^T \mathbf{x} + c$$

với  ${\bf A}$  là một ma trận đối xứng và  ${\bf x}$  là vector có chiều phù hợp.

Nếu  $\bf A$  là một ma trận nửa xác định dương thì  $f({\bf x})$  là một hàm lồi. Nếu  $\bf A$  là một ma trận nửa xác định âm,  $f({\bf x})$  là một hàm lõm.

Nhắc lại hàm mất mát trong hồi quy tuyến tính:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{X}^T \mathbf{w}\|_2^2 = \frac{1}{2N} (\mathbf{y} - \mathbf{X}^T \mathbf{w})^T (\mathbf{y} - \mathbf{X}^T \mathbf{w})$$
$$= \frac{1}{2N} \mathbf{w}^T \mathbf{X} \mathbf{X}^T \mathbf{w} - \frac{1}{N} \mathbf{y}^T \mathbf{X}^T \mathbf{w} + \frac{1}{2N} \mathbf{y}^T \mathbf{y}.$$

Vì  $\mathbf{X}\mathbf{X}^T$  là một ma trận nửa xác định dương, hàm mất mát của hồi quy tuyến tính là một hàm lồi.

$Chu \hat{\tilde{a}} n$

Mọi hàm số bất kỳ thỏa mãn ba điều kiện của chuẩn đều là hàm số lồi. Việc này có thể được trực tiếp suy ra từ bất đẳng thức tam giác của một chuẩn.

Hình 23.10 minh hoạ hai ví dụ về bề mặt của chuẩn  $\ell_1$  và  $\ell_2$  trong không gian hai chiều (chiều thứ ba là giá trị của hàm số). Nhận thấy các bề mặt này đều có một đáy duy nhất tại gốc tọa độ (đây chính là điều kiện đầu tiên của chuẩn).

Hai hàm tiếp theo là ví dụ về các hàm không lồi hoặc lõm. Hàm thứ nhất  $f(x,y)=x^2-y^2$  là một hyperbolic, hàm thứ hai  $f(x,y)=\frac{1}{10}(x^2+2y^2-2\sin(xy))$ . Các bề mặt của hai hàm này được minh họa trên Hình 23.11

### 23.3.4. Đường đồng mức

Để khảo sát tính lồi của các bề mặt trong không gian ba chiều, việc minh hoạ trực tiếp như các ví dụ trên đây có thể khó tưởng tượng hơn. Một phương pháp thường được sử dụng là dùng các đường đồng mức (contour hay level set). Đường đồng mức là cách mô tả các mặt ở không gian ba chiều trong không gian hai chiều. Ở đó, các điểm thuộc cùng một đường tương ứng với các điểm làm cho hàm số có giá trị như nhau. Trong Hình 23.10 và Hình 23.11, các đường nối liền ở mặt phẳng đáy 0xy chính là các đường đồng mức. Nói cách khác, nếu cắt bề mặt bởi các mặt phẳng song song với đáy, ta sẽ thu được các đường đồng mức.

<span id="page-13-0"></span>![](_page_13_Figure_1.jpeg)

Hình 23.10. Ví dụ về mặt của các chuẩn của vector hai chiều.

<span id="page-13-1"></span>![](_page_13_Figure_3.jpeg)

Hình 23.11. Ví dụ về các hàm hai biến không lồi.

Khi khảo sát tính lồi hoặc tìm điểm cực trị của một hàm hai biến, người ta thường vẽ các đường đồng mức thay vì các bề mặt trong không gian ba chiều. Hình 23.12 minh hoạ một vài ví dụ về đường đồng mức. Ở hàng trên, các đường đồng mức là các đường khép kín. Khi các đường này co dần lại ở một điểm thì điểm đó là điểm cực trị. Với các hàm lồi như trong ba ví dụ này, chỉ có một điểm cực trị và đó cũng là điểm cực trị toàn cục. Nếu để ý, bạn sẽ thấy các đường khép kín tạo thành biên của các tập lồi. Ở hàng dưới, các đường không khép kín. Hình 23.12d minh hoạ các đường đồng mức của một hàm tuyến tính f(x,y) = x + y, và là một hàm lồi. Hình 23.12e minh hoạ các đường đồng mức của một hàm lồi (chúng ta sẽ sớm chứng minh) nhưng các đường đồng mức không kín. Hàm này có chứa log nên tập xác định là góc phần tư thứ nhất tương ứng với tọa độ dương (chú ý

<span id="page-14-0"></span>![](_page_14_Figure_1.jpeg)

**Hình 23.12.** Ví dụ về đường đồng mức. Hàng trên: các đường đồng mức càng gần tâm tương ứng với các giá trị càng nhỏ. Hàng dưới: các đường nét đứt tương ứng với các giá trị âm, các đường nét liền tương ứng với các giá trị không âm. Các hàm số đều lồi ngoại trừ hàm số trong hình f).

rằng tập hợp điểm có tọa độ dương cũng là một tập lồi vì nó là một siêu đa diện). Các đường không kín này nếu kết hợp với trục Ox, Oy sẽ tạo thành biên của các tập lồi. Hình 23.12f minh hoạ các đường đồng mức của một hàm hyperbolic, hàm này không phải là một hàm lồi.

### 23.3.5. Tập dưới mức $\alpha$

#### $\overline{\text{Dinh}}$ nghĩa 23.8: Tập dưới mức $\alpha$

Tập dưới mức  $\alpha$  ( $\alpha$ -sublevel set) của một hàm số  $f:\mathbb{R}^n\to\mathbb{R}$  là một tập hợp được định nghĩa bởi

$$\mathcal{C}_{\alpha} = \{ \mathbf{x} \in \mathbf{dom} f \mid f(\mathbf{x}) \le \alpha \}$$

Diễn đạt bằng lời, một tập dưới mức  $\alpha$  của một hàm số f(.) là tập hợp các điểm trong tập xác định của f(.) mà tại đó hàm số đạt giá trị không lớn hơn  $\alpha$ .

<span id="page-15-0"></span>![](_page_15_Figure_1.jpeg)

**Hình 23.13.** Mọi tập dưới mức  $\alpha$  là tập lồi nhưng hàm số là không lồi.

Quay lại với Hình 23.12, hàng trên, tập dưới mức  $\alpha$  là các hình lồi được bao bởi đường đồng mức. Trong Hình 23.12d, tập dưới mức  $\alpha$  là phần nửa mặt phẳng phía dưới xác định bởi các đường thẳng đồng mức. Trong Hình 23.12e, tập dưới mức  $\alpha$  là vùng bị giới hạn bởi các trục tọa độ và các đường đường đồng mức. Trong Hình 23.12f, tập dưới mức  $\alpha$  hơi khó tưởng tượng hơn. Với  $\alpha>0$ , đường đồng mức là các đường nét liền, các tập dưới mức  $\alpha$  tương ứng là phần nằm giữa các đường nét liền này. Có thể nhận thấy các vùng này không phải là tập lồi.

#### $\overline{\text{Dinh lý } 23.2}$

Nếu một hàm số là lồi thì mọi tập dưới mức  $\alpha$  của nó lồi. Điều ngược lại chưa chắc đã đúng, tức nếu các tập dưới mức  $\alpha$  của một hàm số là  $l \hat{o} i$  thì hàm số đó chưa chắc đã  $l \hat{o} i$ .

Điều này chỉ ra rằng nếu tồn tại một giá trị  $\alpha$  sao cho một tập dưới mức  $\alpha$  của một hàm số là không lồi, thì hàm số đó không lồi. Vì vậy, hàm hyperbolic không phải là một hàm lồi. Các ví dụ trong Hình 23.12, trừ Hình 23.12f, đều tương ứng với các hàm lồi.

Xét ví dụ về việc một hàm số không lồi nhưng mọi tập dưới mức  $\alpha$  đều lồi. Hàm  $f(x,y)=-e^{x+y}$  có mọi tập dưới mức  $\alpha$  là một nửa mặt phẳng (lồi), nhưng nó không phải là một hàm lồi (trong trường hợp này nó là một hàm lõm).

Hình 23.13 là một ví dụ khác về việc một hàm số có mọi tập dưới mức  $\alpha$  lồi nhưng không phải là một hàm lồi. Mọi tập dưới mức  $\alpha$ — của hàm số này đều là hình tròn — lồi, nhưng hàm số đó không lồi. Vì có thể tìm được hai điểm trên mặt này sao cho đoạn thẳng nối chúng nằm hoàn toàn phía dưới của mặt. Chẳng hạn, đoạn thẳng nối một điểm ở cánh và một điểm ở đáy không nằm hoàn toàn

phía trên của mặt. Những hàm số có tập xác định là một tập lồi và có mọi tập dưới mức α là lồi được gọi là hàm tựa lồi (quasiconvex function). Mọi hàm lồi đều là tựa lồi nhưng ngược lại không đúng. Định nghĩa chính thức của hàm tựa lồi được phát biểu như sau

#### Định nghĩa 23.9: Hàm tựa lồi

Một hàm số f : C → R với C là một tập con lồi của R <sup>n</sup> được gọi là tựa lồi (quasiconvex) nếu với mọi x, y ∈ C và mọi θ ∈ [0, 1], ta có:

$$f(\theta \mathbf{x} + (1 - \theta)\mathbf{y}) \le \max\{f(\mathbf{x}), f(\mathbf{y})\}\$$

### 23.3.6. Kiểm tra tính chất lồi dựa vào đạo hàm

Ta có thể nhận biết một hàm số khả vi có là hàm lồi hay không dựa vào các đạo hàm bậc nhất hoặc bậc hai của nó. Giả sử rằng các đạo hàm đó tồn tại.

Điều kiện bậc nhất

Trước hết chúng ta định nghĩa phương trình (mặt) tiếp tuyến của một hàm số f khả vi tại một điểm nằm trên đồ thị (mặt) của hàm số đó (x0, f(x0). Với hàm một biến, phương trình tiếp tuyến tại điểm có tọa độ (x0, f(x0)) là

$$y = f'(x_0)(x - x_0) + f(x_0)$$

Với hàm nhiều biến, đặt ∇f(x0) là gradient của hàm số f tại điểm x0, phương trình mặt tiếp tuyến được cho bởi:

$$y = \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + f(\mathbf{x}_0)$$

Điều kiện bậc nhất

Giả sử hàm số f có tập xác định là lồi và có đạo hàm tại mọi điểm trên tập xác định đó. Khi đó, hàm số f lồi khi và chỉ khi với mọi x, x<sup>0</sup> trên tập xác định, ta có:

<span id="page-16-0"></span>
$$f(\mathbf{x}) \ge f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0)$$
 (23.5)

Một hàm số là lồi chặt nếu dấu bằng trong [\(23.5\)](#page-16-0) xảy ra khi và chỉ khi x = x0.

Một cách trực quan hơn, một hàm số là lồi nếu mặt tiếp tuyến tại một điểm bất kỳ không nằm phía trên mặt đồ thị của hàm số đó.

Hình [23.14](#page-17-0) minh hoạ đồ thị của một hàm lồi và một hàm không lồi. Hình [23.14a](#page-17-0) mô tả một hàm lồi. Hình [23.14b](#page-17-0) mô tả một hàm không lồi vì đồ thị của nó không hoàn toàn nằm phía trên đường thẳng tiếp tuyến.

<span id="page-17-0"></span>f khả vi và có tập xác định lồi

![](_page_17_Figure_2.jpeg)

**Hình 23.14.** Kiểm tra tính lồi dựa vào đạo hàm bậc nhất. Trái: hàm lồi vì tiếp tuyến tại mọi điểm đều nằm phía dưới đồ thị của hàm số, phải: hàm không lồi.

 $Vi \ du: f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$  là một hàm lồi nếu  $\mathbf{A}$  là một ma trận nửa xác định dương.

*Chứng minh:* Đạo hàm bậc nhất của  $f(\mathbf{x})$  là  $\nabla f(\mathbf{x}) = 2\mathbf{A}\mathbf{x}$ . Vậy điều kiện bậc nhất có thể viết dưới dạng (chú ý rằng  $\mathbf{A}$  là một ma trận đối xứng):

$$\mathbf{x}^{T}\mathbf{A}\mathbf{x} \geq 2(\mathbf{A}\mathbf{x}_{0})^{T}(\mathbf{x} - \mathbf{x}_{0}) + \mathbf{x}_{0}^{T}\mathbf{A}\mathbf{x}_{0}$$
$$\Leftrightarrow \mathbf{x}^{T}\mathbf{A}\mathbf{x} \geq 2\mathbf{x}_{0}^{T}\mathbf{A}\mathbf{x} - \mathbf{x}_{0}^{T}\mathbf{A}\mathbf{x}_{0}$$
$$\Leftrightarrow (\mathbf{x} - \mathbf{x}_{0})^{T}\mathbf{A}(\mathbf{x} - \mathbf{x}_{0}) \geq 0$$

Bất đẳng thức cuối cùng đúng dựa trên định nghĩa của ma trận nửa xác định dương. Vậy hàm số  $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$  là một hàm lồi.

Điều kiện bậc hai

Với hàm có đối số là một vector có chiều d, đạo hàm bậc nhất của nó là một vector cũng có chiều d. Đạo hàm bậc hai của nó là một ma trận vuông có chiều  $d \times d$ .

Điều kiện bậc hai

Một hàm số f có đạo hàm bậc hai là hàm lỗi nếu tập xác định của nó lồi và Hesse là một ma trận nửa xác định dương với mọi **x** trong tập xác định:

$$\nabla^2 f(\mathbf{x}) \succeq 0.$$

Nếu Hesse của một hàm số là một ma trận xác định dương thì hàm số đó lồi chặt. Tương tự, nếu Hesse là một ma trận xác định âm thì hàm số đó lõm chặt.

Với hàm số một biến f(x), điều kiện này tương đương với  $f''(x) \ge 0$  với mọi x thuộc tập xác định (và tập xác định là lồi).

Ví du:

- Hàm  $f(x) = x \log(x)$  là lồi chặt vì tập xác định x > 0 là một tập lồi và f''(x) = 1/x là một số dương với mọi x thuộc tập xác định.
- Xét hàm số hai biến:  $f(x,y) = x \log(x) + y \log(y)$  trên tập các giá trị dương của x và y. Hàm số này có đạo hàm bậc nhất  $[\log(x) + 1, \log(y) + 1]^T$  và Hesse  $\begin{bmatrix} 1/x & 0 \\ 0 & 1/y \end{bmatrix}$  là một ma trận đường chéo xác định dương. Vậy đây là một hàm lồi chặt.
- Hàm  $f(x) = x^2 + 5\sin(x)$  không là hàm lồi vì đạo hàm bậc hai  $f''(x) = 2 5\sin(x)$  có thể nhận cả giá trị âm và dương.
- Hàm entropy chéo là một hàm lồi chặt. Xét hai xác suất x và 1-x (0 < x < 1), và a là một hằng số thuộc đoạn [0,1]:  $f(x) = -(a\log(x) + (1-a)\log(1-x))$  có đạo hàm bậc hai  $\frac{a}{x^2} + \frac{1-a}{(1-x)^2}$  là một số dương.
- Nếu **A** là một ma trận xác định dương thì  $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T \mathbf{A}\mathbf{x}$  là lồi vì **A** chính là Hesse của nó.

Ngoài ra còn nhiều tính chất thú vị của các hàm lồi, mời bạn đọc thêm Chương 3 của cuốn *Convex Optimization* [BV04].

## 23.4. Tóm tắt

- Machine learning và tối ưu có quan hệ mật thiết với nhau. Trong tối ưu, tối ưu lồi là quan trọng nhất.
- Mọi đoạn thẳng nối hai điểm bất kỳ của một tập lồi nằm trọn vẹn trong tập đó. Giao điểm của các tập lồi tạo thành một tập lồi.
- Một hàm số là lồi nếu đoạn thẳng nối hai điểm bất kỳ trên đồ thì hàm số đó không nằm phía dưới đồ thị đó.
- Một hàm số khả vi là lồi nếu tập xác định của nó là lồi và mặt tiếp tuyến tại một điểm bất kỳ không nằm phía trên đồ thị của hàm số đó.
- $\bullet$  Các chuẩn là các hàm lồi, được sử dụng nhiều trong tối ưu.