# Ôn tập Xác suất

Chương này được viết dựa trên Chương 2 và 3 của cuốn Computer Vision: Models, Learning, and Inference – Simon J.D. Prince (<https://goo.gl/GTEXzd>).

## 3.1. Xác suất

### 3.1.1. Biến ngẫu nhiên

Một biến ngẫu nhiên (random variable) x là một biến dùng để đo những đại lượng không xác định. Biến này có thể được dùng để ký hiệu kết quả/đầu ra của một thí nghiệm, ví dụ như tung đồng xu, hoặc một đại lượng biến đổi trong tự nhiên, ví dụ như nhiệt độ trong ngày. Nếu quan sát một số lượng lớn đầu ra {xi} I <sup>i</sup>=1 của các thí nghiệm này, ta có thể nhận được những giá trị khác nhau ở mỗi thí nghiệm. Tuy nhiên, sẽ có những giá trị xảy ra nhiều lần hơn những giá trị khác, hoặc xảy ra gần một giá trị này hơn những giá trị khác. Thông tin về đầu ra này được đo bởi một phân phối xác suất (probability distribution) được biểu diễn bằng một hàm p(x). Một biến ngẫu nhiên có thể là rời rạc hoặc liên tục.

Một biến ngẫu nhiên rời rạc sẽ lấy giá trị trong một tập hợp các điểm rời rạc cho trước. Ví dụ tung đồng xu thì có hai khả năng là xấp và ngửa. Tập các giá trị này có thể có thứ tự như khi tung xúc xắc hoặc không có thứ tự như khi đầu ra là các giá trị nắng, mưa, bão. Mỗi đầu ra có một giá trị xác suất tương ứng với nó. Các giá trị xác suất này không âm và có tổng bằng một.

Nếu 
$$x$$
 là biến ngẫu nhiên rời rạc thì  $\sum_{x} p(x) = 1.$  (3.1)

Biến ngẫu nhiên liên tục lấy giá trị là các số thực. Những giá trị này có thể là hữu hạn, ví dụ thời gian làm bài của mỗi thí sinh trong một bài thi 180 phút, hoặc vô hạn, ví dụ thời gian phải chờ tới khách hàng tiếp theo. Không như biến ngẫu nhiên rời rạc, xác suất để đầu ra bằng chính xác một giá trị nào đó theo lý thuyết là bằng không. Thay vào đó, phân phối của biến ngẫu nhiên rời rạc thường được xác định dựa trên xác suất để đầu ra rơi vào một khoảng giá trị nào đó. Việc này được mô tả bởi một hàm số được gọi là hàm mật độ xác suất (probability density function, pdf). Hàm mật độ xác suất luôn cho giá trị dương, và tích phân của nó trên toàn miền giá trị đầu ra phải bằng một.

Nếu 
$$x$$
 là biến ngẫu nhiên liên tục thì  $\int p(x)dx = 1.$  (3.2)

Nếu x là một biến ngẫu nhiên rời rạc thì p(x) ≤ 1, ∀x. Trong khi đó, nếu x là biến ngẫu nhiên liên tục, p(x) có thể nhận giá trị không âm bất kỳ, điều này vẫn đảm bảo tích phân của hàm mật độ xác suất theo toàn bộ giá trị của x bằng một.

### 3.1.2. Xác suất đồng thời

Nếu quan sát số lượng lớn các cặp đầu ra của hai biến ngẫu nhiên x và y thì có những cặp đầu ra xảy ra thường xuyên hơn những cặp khác. Thông tin này được biểu diễn bằng một phân phối được gọi là xác suất đồng thời (joint probability) của x và y, được ký hiệu là p(x, y). Hai biến ngẫu nhiên x và y có thể cùng là biến ngẫu nhiên rời rạc, liên tục, hoặc một rời rạc, một liên tục. Luôn nhớ rằng tổng các xác suất trên mọi cặp giá trị (x, y) đều bằng một.

Cả 
$$x$$
 và  $y$  là rời rạc: 
$$\sum_{x,y} p(x,y) = 1. \tag{3.3}$$

Cả 
$$x$$
 và  $y$  là liên tục: 
$$\int p(x,y)dxdy = 1. \tag{3.4}$$

$$x$$
rời rạc,  $y$  liên tục: 
$$\sum_{x} \int p(x,y) dy = \int \left(\sum_{x} p(x,y)\right) dy = 1. \quad (3.5)$$

Xét ví dụ trong Hình [3.1,](#page-2-0) phần "Xác suất đồng thời". Biến ngẫu nhiên x thể hiện điểm thi môn Toán của học sinh ở một trường THPT trong một kỳ thi quốc gia, biến ngẫu nhiên y thể hiện điểm thi môn Vật Lý cũng trong kỳ thi đó. Đại lượng p(x = x ∗ , y = y ∗ ) là tỉ lệ giữa tần suất số học sinh được đồng thời x <sup>∗</sup> điểm môn Toán và y <sup>∗</sup> điểm môn Vật lý với toàn bộ số học sinh của trường đó. Tỉ lệ này có thể coi là xác suất khi số học sinh trong trường là lớn. Ở đây x <sup>∗</sup> và y ∗ là các số xác định. Thông thường, xác suất này được viết gọn lại thành p(x ∗ , y<sup>∗</sup> ), và p(x, y) được dùng như một hàm tổng quát để mô tả các xác suất.

Giả sử thêm rằng điểm các môn là các số tự nhiên từ 1 đến 10.

Các ô vuông màu đen thể hiện xác suất p(x, y), với diện tích ô vuông càng lớn biểu thị xác suất càng cao. Chú ý rằng tổng các xác suất này bằng một.

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

Hình 3.1. Xác suất đồng thời, xác suất biên, xác suất có điền kiện và mối quan hệ giữa chúng

Có thể thấy rằng xác suất để một học sinh được 10 điểm môn Toán và 1 điểm môn Lý rất thấp, điều tương tự xảy ra với 10 điểm môn Lý và 1 điểm môn Toán. Xác suất để một học sinh được khoảng 7 điểm cả hai môn là cao nhất.

Thông thường, chúng ta sẽ làm việc với các bài toán ở đó xác suất được xác định trên nhiều hơn hai biến ngẫu nhiên. Chẳng hạn, p(x, y, z) thể hiện xác suất đồng thời của ba biến ngẫu nhiên x, y và z. Khi có nhiều biến ngẫu nhiên, ta có thể viết chúng dưới dạng vector. Cụ thể, ta có thể viết p(x) để thể hiện xác suất của biến ngẫu nhiên nhiều chiều x = [x1, x2, . . . , xn] T . Khi có nhiều tập các biến ngẫu nhiên, ví dụ x và y, ta có thể viết p(x, y) để thể hiện xác suất đồng thời của tất cả các thành phần trong hai biến ngẫu nhiên nhiều chiều này.

### 3.1.3. Xác suất biên

Nếu biết xác suất đồng thời của nhiều biến ngẫu nhiên, ta cũng có thể xác định được phân phối xác suất của từng biến bằng cách lấy tổng (với biến ngẫu nhiên rời rạc) hoặc tích phân (với biến ngẫu nhiên liên tục) theo tất cả các biến còn lại:

Nếu 
$$x, y$$
 rời rạc:  $p(x) = \sum_{y} p(x, y),$  (3.6)

<span id="page-3-1"></span>
$$p(y) = \sum_{x} p(x, y). \tag{3.7}$$

Nếu 
$$x, y$$
 liên tục:  $p(x) = \int p(x, y) dy$ , (3.8)

<span id="page-3-0"></span>
$$p(y) = \int p(x,y)dx. \tag{3.9}$$

Với nhiều biến hơn, chẳng hạn bốn biến rời rạc x, y, z, w, cách tính được thực hiện tương tự:

$$p(x) = \sum_{y,z,w} p(x,y,z,w),$$
 (3.10)

$$p(x,y) = \sum_{z,w} p(x,y,z,w).$$
 (3.11)

Cách xác định xác suất của một biến dựa trên xác suất đồng thời của nó với các biến khác được gọi là phép biên hoá (marginalization). Xác suất đó được gọi là xác suất biên (marginal probability).

Từ đây trở đi, nếu không đề cập gì thêm, chúng ta sẽ dùng ký hiệu P để chỉ chung cho cả hai loại biến ngẫu nhiên rời rạc và liên tục. Nếu biến là liên tục, ta sẽ ngầm hiểu rằng dấu tổng (P) cần được thay bằng dấu tích phân (R ), biến lấy vi phân chính là biến được viết dưới dấu P. Chẳng hạn, trong [\(3.11\)](#page-3-0), nếu z là liên tục, w là rời rạc, công thức đúng sẽ là

$$p(x,y) = \sum_{w} \left( \int p(x,y,z,w) dz \right) = \int \left( \sum_{w} p(x,y,z,w) \right) dz.$$
 (3.12)

Quay lại ví dụ trong Hình [3.1](#page-2-0) với hai biến ngẫu nhiên rời rạc x, y. Lúc này, p(x) được hiểu là xác suất để một học sinh đạt được x điểm môn Toán. Xác suất này được biểu thị ở khu vực "Xác suất biên". Có hai cách tính xác suất này. Cách thứ nhất là đếm số học sinh được x điểm môn toán rồi chia cho tổng số học sinh. Cách thứ hai dựa trên xác suất đồng thời để một học sinh được x điểm môn Toán và y điểm môn Lý. Số lượng học sinh đạt x = x <sup>∗</sup> điểm môn Toán sẽ bằng tổng số học sinh đạt x = x <sup>∗</sup> điểm môn Toán và y điểm môn Lý, với y là một giá trị bất kỳ từ 1 đến 10. vì vậy, để tính xác suất p(x), ta chỉ cần tính tổng của toàn bộ p(x, y) với y chạy từ 1 đến 10.

Dựa trên nhận xét này, mỗi giá trị của p(x) chính bằng tổng các giá trị trong cột thứ x của hình vuông trung tâm. Mỗi giá trị của p(y) sẽ bằng tổng các giá trị trong hàng thứ y tính từ đưới lên của hình vuông đó. Chú ý rằng tổng các xác suất luôn bằng một.

### 3.1.4. Xác suất có điều kiện.

Dựa vào phân phối điểm của các học sinh, liệu ta có thể tính được xác suất để một học sinh được điểm 10 môn Lý, biết rằng học sinh đó được điểm 1 môn Toán?

Xác suất để một biến ngẫu nhiên x nhận một giá trị nào đó biết rằng biến ngẫu nhiên y có giá trị  $y^*$  được gọi là xác suất có diều kiện (conditional probability), ký hiệu là  $p(x|y=y^*)$ .

Xác suất có điều kiện  $p(x|y=y^*)$  có thể được tính dựa trên xác suất đồng thời p(x,y). Quay lại Hình 3.1 ở khu vực "Xác suất có điều kiện". Nếu biết rằng y=9, xác suất p(x|y=9) có thể tính được dựa trên hàng thứ chín của hình vuông trung tâm, tức hàng p(x,y=9). Xác suất p(x|y=9) lớn nếu p(x,y=9) lớn. Chú ý rằng tổng các xác suất  $\sum_x p(x,y=9)$  nhỏ hơn một, và bằng tổng các xác suất trên hàng thứ chín này. Để thoả mãn điều kiện tổng các xác suất bằng một, ta cần chia mỗi đại lượng p(x,y=9) cho tổng của hàng này, tức là

$$p(x|y=9) = \frac{p(x,y=9)}{\sum_{x} p(x,y=9)} = \frac{p(x,y=9)}{p(y=9)}.$$
 (3.13)

Tổng quát,

$$p(x|y=y^*) = \frac{p(x,y=y^*)}{\sum_{x} p(x,y=y^*)} = \frac{p(x,y=y^*)}{p(y=y^*)},$$
 (3.14)

ở đây ta đã sử dụng công thức tính xác suất biên trong (3.7) cho mẫu số. Thông thường, ta có thể viết xác suất có điều kiện mà không cần chỉ rõ giá trị  $y=y^*$  và có các công thức gọn hơn:

$$p(x|y) = \frac{p(x,y)}{p(y)}, \ p(y|x) = \frac{p(y,x)}{p(x)}.$$
 (3.15)

Từ đó ta có quan hê

<span id="page-4-1"></span>
$$p(x,y) = p(x|y)p(y) = p(y|x)p(x).$$
 (3.16)

Khi có nhiều hơn hai biến ngẫu nhiên, ta có các công thức

<span id="page-4-0"></span>
$$p(x, y, z, w) = p(x, y, z|w)p(w)$$
 (3.17)

$$= p(x, y|z, w)p(z, w) = p(x, y|z, w)p(z|w)p(w)$$
 (3.18)

$$= p(x|y, z, w)p(y|z, w)p(z|w)p(w). (3.19)$$

Công thức (3.19) có dạng chuỗi và được sử dụng nhiều sau này.

### 3.1.5. Quy tắc Bayes

Công thức (3.16) biểu diễn xác suất đồng thời theo hai cách. Từ đó có thể suy ra

$$p(y|x)p(x) = p(x|y)p(y).$$
 (3.20)

Biến đối một chút:

$$p(y|x) = \frac{p(x|y)p(y)}{p(x)}$$
(3.21)

<span id="page-5-1"></span>
$$=\frac{p(x|y)p(y)}{\sum_{y}p(x,y)}$$
(3.22)

<span id="page-5-0"></span>
$$= \frac{p(x|y)p(y)}{\sum\limits_{y} p(x|y)p(y)}.$$
(3.23)

Trong dòng thứ hai và thứ ba, các công thức về xác suất biên và xác suất đồng thời ở mẫu số đã được sử dụng. Từ (3.23) ta có thể thấy rằng p(y|x) hoàn toàn có thể tính được nếu ta biết mọi p(x|y) và p(y). Tuy nhiên, việc tính trực tiếp xác suất này thường phức tạp.

Ba công thức (3.21)-(3.23) thường được gọi là quy tắc Bayes. Chúng được sử dụng rộng rãi trong machine learning

### 3.1.6. Biến ngẫu nhiên độc lập

Nếu biết giá trị của một biến ngẫu nhiên x không mang lại thông tin về việc suy ra giá trị của biến ngẫu nhiên y và ngược lại, thì ta nói rằng hai biến ngẫu nhiên này là  $d\hat{\rho}c$   $l\hat{q}p$ . Chẳng hạn, chiều cao của một học sinh và điểm thi môn Toán của học sinh đó có thể coi là hai biến ngẫu nhiên  $d\hat{\rho}c$  lập. Khi hai biến ngẫu nhiên  $d\hat{\rho}c$  lập, ta có

$$p(x|y) = p(x), (3.24)$$

$$p(y|x) = p(y). (3.25)$$

Thay vào biểu thức xác suất đồng thời trong (3.16), ta có

$$p(x,y) = p(x|y)p(y) = p(x)p(y).$$
 (3.26)

### 3.1.7. Kỳ vọng

 $K\dot{y}\ vong$  (expectation) của một biến ngẫu nhiên x được định nghĩa bởi

$$E[x] = \sum_{x} xp(x) \text{ n\'eu } x \text{ là rời rạc.}$$
 (3.27)

$$E[x] = \int xp(x)dx \text{ n\'eu } x \text{ là liên tục.}$$
 (3.28)

Giả sử f(.) là một hàm số trả về một số với mỗi giá trị x ∗ của biến ngẫu nhiên x. Khi đó, nếu x là biến ngẫu nhiên rời rạc, ta có

$$E[f(x)] = \sum_{x} f(x)p(x). \tag{3.29}$$

Công thức cho biến ngẫu nhiên liên tục cũng được viết tương tự.

Với xác suất đồng thời, kỳ vọng của một hàm cũng được xác định tương tự:

$$E[f(x,y)] = \sum_{x,y} f(x,y)p(x,y)dxdy.$$
(3.30)

Có ba tính chất cần nhớ về kỳ vọng:

a. Kỳ vọng của một hằng số theo một biến ngẫu nhiên x bất kỳ bằng chính hằng số đó:

$$E[\alpha] = \alpha. \tag{3.31}$$

b. Kỳ vọng có tính chất tuyến tính:

$$E[\alpha x] = \alpha E[x], \tag{3.32}$$

$$E[f(x) + g(x)] = E[f(x)] + E[g(x)].$$
 (3.33)

c. Kỳ vọng của tích hai biến ngẫu nhiên độc lập bằng tích kỳ vọng của chúng:

$$E[f(x)g(y)] = E[f(x)]E[g(y)]. \tag{3.34}$$

Khái niệm kỳ vọng thường đi kèm với khái niệm phương sai (variance) trong không gian một chiều và ma trận hiệp phương sai (covariance matrix) trong không gian nhiều chiều.

### 3.1.8. Phương sai

Cho N giá trị x1, x2, . . . , x<sup>N</sup> . Kỳ vọng và phương sai của bộ dữ liệu này được tính theo công thức

$$\bar{x} = \frac{1}{N} \sum_{n=1}^{N} x_n = \frac{1}{N} \mathbf{x} \mathbf{1},$$
 (3.35)

$$\sigma^2 = \frac{1}{N} \sum_{n=1}^{N} (x_n - \bar{x})^2, \tag{3.36}$$

với x = x1, x2, . . . , x<sup>N</sup> , và 1 ∈ R <sup>N</sup> là vector cột chứa toàn phần tử 1. Kỳ vọng đơn giản là trung bình cộng của toàn bộ các giá trị. Phương sai là trung bình

<span id="page-7-0"></span>![](_page_7_Picture_1.jpeg)

Hình 3.2. Ví dụ về kỳ vọng và phương sai. (a) Dữ liệu trong không gian một chiều. (b) Dữ liệu trong không gian hai chiều mà hai chiều không tương quan. Trong trường hợp này, ma trận hiệp phương sai là ma trận đường chéo với hai phần tử trên đường chéo là σ1, σ2, đây cũng chính là hai trị riêng của ma trận hiệp phương sai và là phương sai của mỗi chiều dữ liệu. (c) Dữ liệu trong không gian hai chiều có tương quan. Theo mỗi chiều, ta có thể tính được kỳ vọng và phương sai. Phương sai càng lớn thì dữ liệu trong chiều đó càng phân tán. Trong ví dụ này, dữ liệu theo chiều thứ hai phân tán nhiều hơn so với chiều thứ nhất.

cộng của bình phương khoảng cách từ mỗi điểm tới kỳ vọng. Phương sai càng nhỏ, các điểm dữ liệu càng gần với kỳ vọng, tức các điểm dữ liệu càng giống nhau. Phương sai càng lớn, dữ liệu càng có tính phân tán. Ví dụ về kỳ vọng và phương sai của dữ liệu một chiều có thể được thấy trong Hình [3.2a.](#page-7-0)

Căn bậc hai của phương sai, σ còn được gọi là độ lệch chuẩn (standard deviation) của dữ liệu.

### 3.1.9. Ma trận hiệp phương sai

Cho N điểm dữ liệu được biểu diễn bởi các vector cột x1, . . . , x<sup>N</sup> , khi đó, vector kỳ vọng và ma trận hiệp phương sai của toàn bộ dữ liệu được định nghĩa là

$$\bar{\mathbf{x}} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n, \tag{3.37}$$

$$\mathbf{S} = \frac{1}{N} \sum_{n=1}^{N} (\mathbf{x}_n - \bar{\mathbf{x}}) (\mathbf{x}_n - \bar{\mathbf{x}})^T = \frac{1}{N} \hat{\mathbf{X}} \hat{\mathbf{X}}^T.$$
 (3.38)

Trong đó Xˆ được tạo bằng cách trừ mỗi cột của X đi x¯:

$$\hat{\mathbf{x}}_n = \mathbf{x}_n - \bar{\mathbf{x}}.\tag{3.39}$$

Một vài tính chất của ma trận hiệp phương sai:

- a. Ma trận hiệp phương sai là một ma trận đối xứng, hơn nữa, nó là một ma trận [nửa xác định dương.](https://machinelearningcoban.com/2017/03/12/convexity/#positive-semidefinite)
- b. Mọi phần tử trên đường chéo của ma trận hiệp phương sai là các số không âm. Chúng chính là phương sai của từng chiều dữ liệu.
- c. Các phần tử ngoài đường chéo sij , i 6= j thể hiện sự tương quan giữa thành phần thứ i và thứ j của dữ liệu, còn được gọi là hiệp phương sai. Giá trị này có thể dương, âm hoặc bằng không. Khi nó bằng không, ta nói rằng hai thành phần i, j trong dữ liệu là không tương quan.
- d. Nếu ma trận hiệp phương sai là ma trận đường chéo, ta có dữ liệu hoàn toàn không tương quan giữa các chiều.

Ví dụ về sự tương quan của dữ liệu được cho trong Hình [3.2b](#page-7-0) và [3.2c.](#page-7-0)

## 3.2. Một vài phân phối thường gặp

### 3.2.1. Phân phối Bernoulli

Phân phối Bernoulli là một phân phối rời rạc mô tả các biến ngẫu nhiên nhị phân với đầu ra chỉ nhận một trong hai giá trị x ∈ {0, 1}. Hai giá trị này có thể là xấp và ngửa khi tung đồng xu; có thể là giao dịch lừa đảo và giao dịch thông thường trong bài toán xác định giao dịch lừa đảo trong tín dụng; có thể là người và không phải người trong bài toán xác định xem trong một bức ảnh có người hay không.

Phân phối Bernoulli được mô tả bằng một tham số λ ∈ [0, 1]. Xác suất của mỗi đầu ra là

$$p(x=1) = \lambda, \quad p(x=0) = 1 - p(x=1) = 1 - \lambda.$$
 (3.40)

Hai đẳng thức này thường được viết gọn lại thành

$$p(x) = \lambda^x (1 - \lambda)^{1 - x}, \tag{3.41}$$

với giả định 0 <sup>0</sup> = 1. Thật vậy, p(0) = λ 0 (1−λ) <sup>1</sup> = 1−λ, và p(1) = λ 1 (1−λ) <sup>0</sup> = λ.

Phân phối Bernoulli thường được ký hiệu ngắn gọn dưới dạng

$$p(x) = \operatorname{Bern}_{x}[\lambda]. \tag{3.42}$$

### 3.2.2. Phân phối categorical

Trong nhiều trường hợp, đầu ra của biến ngẫu nhiên rời rạc có thể nhận nhiều hơn hai giá trị. Ví dụ, một bức ảnh có thể chứa một chiếc xe, một người, hoặc một con mèo. Khi đó, ta dùng một phân phối tổng quát của phân phối Bernoulli, được gọi là phân phối categorical. Các đầu ra được mô tả bởi một phần tử trong tập hợp  $\{1,2,\ldots,K\}$ .

Nếu có K đầu ra, phân phối categorical sẽ được mô tả bởi K tham số, viết dưới dạng vector  $\lambda = [\lambda_1, \lambda_2, \dots, \lambda_K]$  với các  $\lambda_k$  không âm và có tổng bằng một. Mỗi giá trị  $\lambda_k$  thể hiện xác suất để đầu ra nhận giá trị k:  $p(x = k) = \lambda_k$ .

Phân phối categorical thường được ký hiệu dưới dạng:

$$p(x) = \operatorname{Cat}_x[\lambda]. \tag{3.43}$$

Cách biểu diễn đầu ra là một số k trong tập hợp  $\{1, 2, ..., K\}$  có thể được thay bằng biểu diễn *one-hot*. Mỗi vector one-hot là một vector K phần tử, trong đó K-1 phần tử bằng 0, một phần tử bằng 1 tại vị trí ứng với đầu ra k. Nói cách khác, mỗi đầu ra là một trong các vector đơn vị bậc K:  $\{\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_K\}$ . Ta có thể viết

$$p(x=k) = p(\mathbf{x} = \mathbf{e}_k) = \prod_{j=1}^K \lambda_j^{x_j} = \lambda_k.$$
 (3.44)

Dấu bằng cuối cùng xảy ra vì  $x_k = 1, x_j = 0 \ \forall j \neq k.$ 

### 3.2.3. Phân phối chuẩn một chiều

Phân phối chuẩn một chiều (univariate normal distribution) được định nghĩa trên các biến liên tục nhận giá trị  $x \in (-\infty, \infty)$ . Đây là một phân phối được sử dụng nhiều nhất với các biến ngẫu nhiên liên tục. Phân phối này được mô tả bởi hai tham số: kỳ vọng  $\mu$  và phương sai  $\sigma^2$ .

Hàm mật độ xác suất của phân phối này được định nghĩa bởi

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right). \tag{3.45}$$

Hàm mật độ này thường được viết gọn dưới dạng  $p(x)=\mathrm{Norm}_x[\mu,\sigma^2]$  hoặc  $\mathcal{N}(\mu,\sigma^2)$ .

Ví dụ về đồ thị hàm mật độ xác suất của phân phối chuẩn một chiều được biểu thị trên Hình 3.3a.

### 3.2.4. Phân phối chuẩn nhiều chiều

Phân phối chuẩn nhiều chiều (multivariate normal distribution) là trường hợp tổng quát của phân phối chuẩn khi biến ngẫu nhiên là nhiều chiều, giả sử là D chiều. Có hai tham số mô tả phân phối này là vector kỳ vọng  $\boldsymbol{\mu} \in \mathbb{R}^D$  và ma trận hiệp phương sai  $\boldsymbol{\Sigma} \in \mathbb{S}^D$  là một ma trận đối xứng xác định dương.

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

Hình 3.3. Ví dụ về hàm mật độ xác suất của (a) phân phối chuẩn một chiều, và (b) phân phối chuẩn hai chiều.

Hàm mật độ xác suất có dạng

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2} |\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right), \quad (3.46)$$

với |Σ| là định thức của ma trận hiệp phương sai Σ.

Hàm mật độ này thường được viết gọn lại dưới dạng p(x) = Normx[µ, Σ] hoặc N (µ, Σ).

Ví dụ về hàm mật độ xác suất của một phân phối chuẩn hai chiều được mô tả bởi một mặt cong trên Hình [3.3b.](#page-10-0) Nếu cắt mặt này theo các mặt phẳng song song với mặt đáy, ta sẽ thu được các hình ellipse đồng tâm.

### 3.2.5. Phân phối Beta

Phân phối Beta là một phân phối liên tục được định nghĩa trên một biến ngẫu nhiên λ ∈ [0, 1]. Phân phối Beta được dùng để mô tả tham số cho một phân phối khác. Cụ thể, phân phối này phù hợp với việc mô tả sự biến động của tham số λ trong phân phối Bernoulli.

Phân phối Beta được mô tả bởi hai tham số dương α, β. Hàm mật độ xác suất của nó được cho bởi

<span id="page-10-1"></span>
$$p(\lambda) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \lambda^{\alpha - 1} (1 - \lambda)^{\beta - 1}, \tag{3.47}$$

với Γ(.) là hàm số gamma, được định nghĩa bởi

$$\Gamma(z) = \int_0^\infty t^{z-1} \exp(-t) dt.$$
 (3.48)

<span id="page-11-0"></span>![](_page_11_Figure_1.jpeg)

Hình 3.4. Ví dụ về hàm mật độ xác suất của phân phối Beta. (a) α = β, đồ thị hàm số là đối xứng. (b) α < β, đồ thị hàm số lệch sang trái, chứng tỏ xác suất λ nhỏ là lớn. (c) α > β, đồ thị hàm số lệch sang phải, chứng tỏ xác suất λ lớn là lớn.

Trên thực tế, việc tính giá trị của hàm số gamma không thực sự quan trọng vì nó chỉ mang tính chuẩn hoá để tổng xác suất bằng một.

Phân phối Beta thường được ký hiệu là p(λ) = Betaλ[α, β].

Hình [3.4](#page-11-0) minh hoạ hàm mật độ xác suất của phân phối Beta với các cặp giá trị (α, β) khác nhau.

- Trong Hình [3.4a,](#page-11-0) khi α = β. Đồ thị của các hàm mật độ xác suất đối xứng qua đường thẳng λ = 0.5. Khi α = β = 1, thay vào [\(3.47\)](#page-10-1), ta thấy p(λ) = 1 với mọi λ. Trong trường hợp này, phân phối Beta trở thành phân phối đều Khi α = β > 1, các hàm số đạt giá trị cao tại gần trung tâm, tức λ sẽ nhận giá trị xung quanh điểm 0.5 với xác suất cao. Khi α = β < 1, hàm số đạt giá trị cao tại các điểm gần 0 và 1.
- Trong Hình [3.4b,](#page-11-0) khi α < β, ta thấy rằng đồ thị có xu hướng lệch sang bên trái. Các giá trị (α, β) này nên được sử dụng nếu ta dự đoán rằng λ là một số nhỏ hơn 0.5.
- Trong Hình [3.4c,](#page-11-0) khi α > β, điều ngược lại xảy ra với các hàm số đạt giá trị cao tại các điểm gần 1.

### 3.2.6. Phân phối Dirichlet

Phân phối Dirichlet chính là trường hợp tổng quát của phân phối Beta khi được dùng để mô tả tham số của phân phối categorical. Nhắc lại rằng phân phối categorical là trường hợp tổng quát của phân phối Bernoulli.

Phân phối Dirichlet được định nghĩa trên K biến liên tục  $\lambda_1, \ldots, \lambda_K$  trong đó các  $\lambda_k$  không âm và có tổng bằng một. Bởi vậy, nó phù hợp để mô tả tham số của phân phối categorical. Có K tham số dương để mô tả một phân phối Dirichlet:  $\alpha_1, \ldots, \alpha_K$ .

Hàm mật độ xác suất của phân phối Dirichlet được cho bởi

$$p(\lambda_1, \dots, \lambda_K) = \frac{\Gamma(\sum_{k=1}^K \alpha_k)}{\prod_{k=1}^K \Gamma(\alpha_k)} \prod_{k=1}^K \lambda_k^{\alpha_k - 1}.$$
 (3.49)

Dạng thu gọn của nó là  $p(\lambda_1,\ldots,\lambda_K)=\mathrm{Dir}_{\lambda_1,\ldots,\lambda_K}[\alpha_1,\ldots,\alpha_K].$