## <span id="page-0-4"></span>Phương pháp nhân tử Lagrange

Việc tối ưu hàm số một biến liên tục và khả vi trên miền xác định là một tập mở¹ thường được thực hiện dựa trên việc giải phương trình đạo hàm bằng không. Gọi hàm mục tiêu là  $f(x): \mathbb{R} \to \mathbb{R}$ , cực trị toàn cục nếu có thường được tìm bằng cách giải phương trình f'(x) = 0. Chú ý rằng điều ngược lại không đúng, tức một điểm thoả mãn đạo hàm bằng không chưa chắc đã là cực trị của hàm số. Ví dụ hàm  $f(x) = x^3$  có đạo hàm bằng không tại x = 0 nhưng điểm này không là một điểm cực trị. Với hàm nhiều biến, ta cũng có thể áp dụng quan sát này: giải phương trình gradient bằng không.

Cách làm trên đây được áp dụng vào các bài toán tối ưu không ràng buộc. Các bài toán có ràng buộc là một phương trình:

<span id="page-0-1"></span>
$$\mathbf{x} = \arg\min_{\mathbf{x}} f_0(\mathbf{x})$$
thoả mãn:  $f_1(\mathbf{x}) = 0$ , (A.1)

cũng có thể được đưa về bài toán không ràng buộc bằng *phương pháp nhân tử* Lagrange.

Xét hàm số  $\mathcal{L}(\mathbf{x},\lambda) = f_0(\mathbf{x}) + \lambda f_1(\mathbf{x})$  với biến  $\lambda$  được gọi là *nhân tử Lagrange* (Lagrange multiplier). Hàm số  $\mathcal{L}(\mathbf{x},\lambda)$  được gọi là *hàm Lagrange* của bài toán. Người ta đã chứng minh được rằng, điểm tối ưu của bài toán (A.1) thoả mãn điều kiện  $\nabla_{\mathbf{x},\lambda}\mathcal{L}(\mathbf{x},\lambda) = \mathbf{0}$ . Điều này tương đương với:

<span id="page-0-2"></span>
$$\nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda) = \nabla_{\mathbf{x}} f_0(\mathbf{x}) + \lambda \nabla_{\mathbf{x}} f_1(\mathbf{x}) = 0$$
(A.2)

<span id="page-0-3"></span>
$$\nabla_{\lambda} \mathcal{L}(\mathbf{x}, \lambda) = f_1(\mathbf{x}) = 0 \tag{A.3}$$

 $Dể {y}$  rằng điều kiện thứ hai chính là ràng buộc trong bài toán (A.1).

<span id="page-0-0"></span><sup>1</sup> Xem thêm: Open sets, closed sets and sequences of real numbers (https://goo.gl/AgKhCn).

Trong nhiều trường hợp, việc giải hệ phương trình (A.2) - (A.3) đơn giản hơn việc trực tiếp đi tìm nghiệm của bài toán (A.1).

#### Ví dụ 1:

Tìm giá trị lớn nhất và nhỏ nhất của hàm số  $f_0(x,y) = x + y$  với x,y thoả mãn điều kiện  $f_1(x,y) = x^2 + y^2 = 2$ .

#### Lời giải:

Điều kiện ràng buộc có thể được viết lại dưới dạng  $x^2+y^2-2=0$ . Hàm Lagrange của bài toán này là  $\mathcal{L}(x,y,\lambda)=x+y+\lambda(x^2+y^2-2)$ . Các điểm cực trị của hàm số Lagrange phải thoả mãn hệ điều kiện:

<span id="page-1-0"></span>
$$\nabla_{x,y,\lambda} \mathcal{L}(x,y,\lambda) = 0 \Leftrightarrow \begin{cases} 1 + 2\lambda x = 0\\ 1 + 2\lambda y = 0\\ x^2 + y^2 = 2 \end{cases}$$
(A.4)

Từ hai phương trình đầu của (A.4) suy ra  $x = y = \frac{-1}{2\lambda}$ . Thay vào phương trình cuối ta sẽ có  $\lambda^2 = \frac{1}{4} \Rightarrow \lambda = \pm \frac{1}{2}$ . Vậy ta được 2 cặp nghiệm  $(x,y) \in \{(1,1),(-1,-1)\}$ . Bằng cách thay các giá trị này vào hàm mục tiêu, ta tìm được giá trị nhỏ nhất và lớn nhất của bài toán.

#### **Ví dụ 2**: Chuẩn $\ell_2$ của ma trận.

Nhắc lại chuẩn  $\ell_2$  của một vector  $\mathbf{x} : \|\mathbf{x}\|_2 = \sqrt{\mathbf{x}^T \mathbf{x}}$ . Dựa trên chuẩn  $\ell_2$  của vector, chuẩn  $\ell_2$  của một ma trận  $\mathbf{A} \in \mathbb{R}^{m \times n}$ , ký hiệu là  $\|\mathbf{A}\|_2$ , được định nghĩa như sau:

$$\|\mathbf{A}\|_{2} = \max \frac{\|\mathbf{A}\mathbf{x}\|_{2}}{\|\mathbf{x}\|_{2}} = \max \sqrt{\frac{\mathbf{x}^{T}\mathbf{A}^{T}\mathbf{A}\mathbf{x}}{\mathbf{x}^{T}\mathbf{x}}}, \text{v\'oi } \mathbf{x} \in \mathbb{R}^{n}$$
 (A.5)

Bài toán tối ưu này tương đương với:

$$\max (\mathbf{x}^T \mathbf{A}^T \mathbf{A} \mathbf{x})$$
thoả mãn:  $\mathbf{x}^T \mathbf{x} = 1$  (A.6)

Hàm Lagrange của bài toán này là

$$\mathcal{L}(\mathbf{x}, \lambda) = \mathbf{x}^T \mathbf{A}^T \mathbf{A} \mathbf{x} + \lambda (1 - \mathbf{x}^T \mathbf{x})$$
(A.7)

Các điểm cực trị của hàm số Lagrange phải thoả mãn:

<span id="page-1-1"></span>
$$\nabla_{\mathbf{x}} \mathcal{L} = 2\mathbf{A}^T \mathbf{A} \mathbf{x} - 2\lambda \mathbf{x} = \mathbf{0}$$
 (A.8)

$$\nabla_{\lambda} \mathcal{L} = 1 - \mathbf{x}^T \mathbf{x} = 0 \tag{A.9}$$

Từ (A.8) ta có  $\mathbf{A}^T \mathbf{A} \mathbf{x} = \lambda \mathbf{x}$ . Vậy  $\mathbf{x}$  phải là một vector riêng của  $\mathbf{A}^T \mathbf{A}$  ứng với trị riêng  $\lambda$ . Nhân cả hai vế của biểu thức này với  $\mathbf{x}^T$  vào bên trái và sử dụng (A.9), ta thu được:

$$\mathbf{x}^T \mathbf{A}^T \mathbf{A} \mathbf{x} = \lambda \mathbf{x}^T \mathbf{x} = \lambda \tag{A.10}$$

Từ đó suy ra kAxk<sup>2</sup> đạt giá trị lớn nhất khi λ đạt giá trị lớn nhất. Nói cách khác, λ phải là trị riêng lớn nhất của A<sup>T</sup>A. Vậy, kAk<sup>2</sup> = λmax(A<sup>T</sup>A).

Các trị riêng của A<sup>T</sup>A còn được gọi là giá trị suy biến (singular value) của A. Tóm lại, chuẩn `<sup>2</sup> của một ma trận là giá trị suy biến lớn nhất của ma trận đó.

Hoàn toàn tương tự, nghiệm của bài toán

$$\min_{\|\mathbf{x}\|=1} \|\mathbf{A}\mathbf{x}\|_2 \tag{A.11}$$

chính là một vector riêng ứng với giá trị suy biến nhỏ nhất của A.

# Ảnh màu

<span id="page-3-0"></span>![](_page_3_Picture_3.jpeg)

Hình B.1. Ví dụ về 1NN. Các hình tròn khác màu thể hiện các điểm dữ liệu huấn luyện của các lớp khác nhau. Các vùng nền thể hiện những điểm được phân loại vào lớp có màu tương ứng khi sử dụng 1NN (Nguồn: K-nearest neighbors algorithm – Wikipedia [\(https://goo.gl/](https://goo.gl/Ba4xhX) [Ba4xhX\)](https://goo.gl/Ba4xhX), ảnh màu của Hình [B.1.](#page-3-0)).

![](_page_3_Picture_5.jpeg)

Hình B.2. Ba loại hoa lan trong bộ cơ sở dữ liệu hoa Iris (ảnh màu của Hình 9.2).

![](_page_4_Picture_1.jpeg)

Hình B.3. Ảnh:Trọng Vũ [\(https:](https://goo.gl/9D8aXW) [//goo.gl/9D8aXW,](https://goo.gl/9D8aXW) ảnh màu của Hình 10.7) .

![](_page_4_Picture_3.jpeg)

Hình B.4. Kết quả nhận được sau khi thực hiện phân cụm K-means cho các điểm dữ liệu. Có ba cụm dữ liệu tương ứng với ba màu đỏ, hồng, đen (ảnh màu của Hình 10.8).

![](_page_4_Picture_5.jpeg)

![](_page_4_Picture_6.jpeg)

![](_page_4_Picture_8.jpeg)

![](_page_4_Picture_9.jpeg)

Hình B.5. Chất lượng nén ảnh với số lượng cụm khác nhau (ảnh màu của Hình 10.9).

![](_page_5_Figure_1.jpeg)

Hình B.6. Đồ thị hàm sigmoid trong không gian hai chiều (ảnh màu của Hình 14.4b).

![](_page_5_Figure_3.jpeg)

Hình B.7. Ví dụ về các bức ảnh trong 10 lớp của bộ dữ liệu CIFAR10 (ảnh màu của Hình 29.1).

<span id="page-5-0"></span>![](_page_5_Figure_5.jpeg)

Hình B.8. Minh họa hệ số tìm được dưới dạng các bức ảnh (ảnh màu của Hình [B.8\)](#page-5-0).

### Tài liệu tham khảo

- [AKA91] David W Aha, Dennis Kibler, and Marc K Albert. Instance-based learning algorithms. Machine learning, 6(1):37–66, 1991.
- [AM93] Sunil Arya and David M Mount. Algorithms for fast vector quantization. In Data Compression Conference, pages 381–390. IEEE, 1993.
- [AMMIL12] Yaser S Abu-Mostafa, Malik Magdon-Ismail, and Hsuan-Tien Lin. Learning from data, volume 4. AMLBook New York, NY, USA:, 2012.
  - [AV07] David Arthur and Sergei Vassilvitskii. K-means++: The advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms, pages 1027–1035. Society for Industrial and Applied Mathematics, 2007.
  - [Bis06] Christopher M Bishop. Pattern recognition and machine learning. Springer, 2006.
  - [BL14] Artem Babenko and Victor Lempitsky. Additive quantization for extreme vector compression. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition, pages 931–938, 2014.
  - [Ble08] David M Blei. Hierarchical clustering. 2008.
  - [BMV<sup>+</sup>12] Bahman Bahmani, Benjamin Moseley, Andrea Vattani, Ravi Kumar, and Sergei Vassilvitskii. Scalable k-means++. Proceedings of the VLDB Endowment, 5(7):622–633, 2012.
  - [BTVG06] Herbert Bay, Tinne Tuytelaars, and Luc Van Gool. SURF: Speeded Up Robust Features. Proceedings IEEE European Conference on Computer Vision, pages 404–417, 2006.
    - [BV04] Stephen Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.
- [CLMW11] Emmanuel J Candès, Xiaodong Li, Yi Ma, and John Wright. Robust principal component analysis? Journal of the ACM (JACM),

- 58(3):11, 2011.
- [Cyb89] George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems (MCSS), 2(4):303–314, 1989.
- [DFK<sup>+</sup>04] Petros Drineas, Alan Frieze, Ravi Kannan, Santosh Vempala, and V Vinay. Clustering large graphs via the singular value decomposition. Machine learning, 56(1):9–33, 2004.
- [dGJL05] Alexandre d'Aspremont, Laurent E Ghaoui, Michael I Jordan, and Gert R Lanckriet. A direct formulation for sparse pca using semidefinite programming. In Advances in Neural Information Processing Systems, pages 41–48, 2005.
- [DHS11] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(Jul):2121–2159, 2011.
- [DT05] Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition, volume 1, pages 886–893. IEEE, 2005.
- [ERK<sup>+</sup>11] Michael D Ekstrand, John T Riedl, Joseph A Konstan, et al. Collaborative filtering recommender systems. Foundations and Trends® in Human–Computer Interaction, 4(2):81–173, 2011.
  - [FHT01] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. The elements of statistical learning, volume 1. Springer series in statistics New York, 2001.
  - [Fuk13] Keinosuke Fukunaga. Introduction to statistical pattern recognition. Academic press, 2013.
  - [GBC16] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016. <http://www.deeplearningbook.org>.
    - [GR70] Gene H Golub and Christian Reinsch. Singular value decomposition and least squares solutions. Numerische mathematik, 14(5):403–420, 1970.
- [HNO06] Per Christian Hansen, James G Nagy, and Dianne P O'leary. Deblurring images: matrices, spectra, and filtering. SIAM, 2006.
- [HZRS16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.
  - [JDJ17] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. arXiv preprint arXiv:1702.08734, 2017.

- [JDS11] Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1):117–128, 2011.
- [KA04] Shehroz S Khan and Amir Ahmad. Cluster center initialization algorithm for k-means clustering. Pattern recognition letters, 25(11):1293–1302, 2004.
- [KB14] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [KBV09] Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8), 2009.
  - [KH92] Anders Krogh and John A Hertz. A simple weight decay can improve generalization. In Advances in Neural Information Processing Systems, pages 950–957, 1992.
- [KSH12] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, pages 1097–1105, 2012.
- [LCB10] Yann LeCun, Corinna Cortes, and Christopher JC Burges. Mnist handwritten digit database. AT&T Labs [Online]. Available: http://yann. lecun. com/exdb/mnist, 2, 2010.
- [LCD04] Anukool Lakhina, Mark Crovella, and Christophe Diot. Diagnosing network-wide traffic anomalies. In ACM SIGCOMM Computer Communication Review, volume 34, pages 219–230. ACM, 2004.
- [Low99] David G Lowe. Object recognition from local scale-invariant features. In Proceedings IEEE International Conference on Computer Vision, volume 2, pages 1150–1157. IEEE, 1999.
- [LSP06] Svetlana Lazebnik, Cordelia Schmid, and Jean Ponce. Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition, volume 2, pages 2169–2178, 2006.
- [LW<sup>+</sup>02] Andy Liaw, Matthew Wiener, et al. Classification and regression by randomforest. R news, 2(3):18–22, 2002.
  - [M<sup>+</sup>97] Tom M Mitchell et al. Machine learning. wcb, 1997.
- [MSS<sup>+</sup>99] Sebastian Mika, Bernhard Sch¨olkopf, Alex J Smola, Klaus-Robert M¨uller, Matthias Scholz, and Gunnar R¨atsch. Kernel pca and denoising in feature spaces. In Advances in Neural Information Processing Systems, pages 536–542, 1999.
  - [Nes07] Yurii Nesterov. Gradient methods for minimizing composite objective function, 2007.

- [NF13] Mohammad Norouzi and David J Fleet. Cartesian k-means. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition, pages 3017–3024, 2013.
- [NJW02] Andrew Y Ng, Michael I Jordan, and Yair Weiss. On spectral clustering: Analysis and an algorithm. In Advances in Neural Information Processing Systems, pages 849–856, 2002.
  - [Pat07] Arkadiusz Paterek. Improving regularized singular value decomposition for collaborative filtering. In Proceedings of KDD cup and workshop, volume 2007, pages 5–8, 2007.
  - [Pla98] John Platt. Sequential minimal optimization: A fast algorithm for training support vector machines. 1998.
  - [Pri12] Simon JD Prince. Computer vision: models, learning, and inference. Cambridge University Press, 2012.
- [RDVC<sup>+</sup>04] Lorenzo Rosasco, Ernesto De Vito, Andrea Caponnetto, Michele Piana, and Alessandro Verri. Are loss functions all the same? Neural Computation, 16(5):1063–1076, 2004.
  - [Rey15] Douglas Reynolds. Gaussian mixture models. Encyclopedia of biometrics, pages 827–832, 2015.
  - [Ros57] F Rosemblat. The perceptron: A perceiving and recognizing automation. Cornell Aeronautical Laboratory Report, 1957.
  - [Rud16] Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016.
  - [SCSC03] Mei-Ling Shyu, Shu-Ching Chen, Kanoksri Sarinnapakorn, and LiWu Chang. A novel anomaly detection scheme based on principal component classifier. Technical report, MIAMI UNIV CORAL GABLES FL DEPT OF ELECTRICAL AND COMPUTER ENGINEERING, 2003.
  - [SFHS07] J Ben Schafer, Dan Frankowski, Jon Herlocker, and Shilad Sen. Collaborative filtering recommender systems. In The adaptive web, pages 291–324. Springer, 2007.
  - [SHK<sup>+</sup>14] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.
  - [SKKR00] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. Application of dimensionality reduction in recommender system-a case study. Technical report, Minnesota Univ Minneapolis Dept of Computer Science, 2000.

- [SKKR02] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. Incremental singular value decomposition algorithms for highly scalable recommender systems. In Fifth International Conference on Computer and Information Science, pages 27–28. Citeseer, 2002.
  - [SLJ<sup>+</sup>15] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015.
- [SSWB00] Bernhard Sch¨olkopf, Alex J Smola, Robert C Williamson, and Peter L Bartlett. New support vector algorithms. Neural computation, 12(5):1207–1245, 2000.
- [SWY75] Gerard Salton, Anita Wong, and Chung-Shu Yang. A vector space model for automatic indexing. Communications of the ACM, 18(11):613–620, 1975.
  - [SZ14] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
  - [TH12] Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURS-ERA: Neural networks for machine learning, 4(2):26–31, 2012.
- [VJG14] João Vinagre, Alípio Mário Jorge, and João Gama. Fast incremental matrix factorization for recommendation with positive-only feedback. In International Conference on User Modeling, Adaptation, and Personalization, pages 459–470. Springer, 2014.
  - [VL07] Ulrike Von Luxburg. A tutorial on spectral clustering. Statistics and computing, 17(4):395–416, 2007.
- [VM16] Tiep Vu and Vishal Monga. Learning a low-rank shared dictionary for object classification. In Proceedings IEEE International Conference on Image Processing, pages 4428–4432. IEEE, 2016.
- [VM17] Tiep Vu and Vishal Monga. Fast low-rank shared dictionary learning for image classification. IEEE Transactions on Image Processing, 26(11):5160–5175, Nov 2017.
- [VMM<sup>+</sup>16] Tiep Vu, Hojjat Seyed Mousavi, Vishal Monga, Ganesh Rao, and UK Arvind Rao. Histopathological image classification using discriminative feature-oriented dictionary learning. IEEE Transactions on Medical Imaging, 35(3):738–751, 2016.
- [WYG<sup>+</sup>09] John Wright, Allen Y Yang, Arvind Ganesh, S Shankar Sastry, and Yi Ma. Robust face recognition via sparse representation. IEEE Transactions on Pattern Analysis and Machine Intelligence,

- 31(2):210–227, 2009.
- [XWCL15] Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li. Empirical evaluation of rectified activations in convolutional network. arXiv preprint arXiv:1505.00853, 2015.
- [YZFZ11] M. Yang, L. Zhang, X. Feng, and D. Zhang. Fisher discrimination dictionary learning for sparse representation. In Proceedings IEEE International Conference on Computer Vision, pages 543–550, Nov. 2011.
- [ZDW14] Ting Zhang, Chao Du, and Jingdong Wang. Composite quantization for approximate nearest neighbor search. In International Conference on Machine Learning, number 2, pages 838–846, 2014.
  - [ZF14] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In Proceedings IEEE European Conference on Computer Vision, pages 818–833. Springer, 2014.
- [ZWFM06] Sheng Zhang, Weihong Wang, James Ford, and Fillia Makedon. Learning from incomplete ratings using non-negative matrix factorization. In Proceedings of the 2006 SIAM International Conference on Data Mining, pages 549–553. SIAM, 2006.
  - [ZYK06] Haitao Zhao, Pong Chi Yuen, and James T Kwok. A novel incremental principal component analysis and its application for face recognition. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 36(4):873–886, 2006.
- [ZYX<sup>+</sup>08] Zhi-Qiang Zeng, Hong-Bin Yu, Hua-Rong Xu, Yan-Qi Xie, and Ji Gao. Fast training support vector machines using parallel sequential minimal optimization. In International Conference on Intelligent System and Knowledge Engineering, volume 1, pages 997–1001. IEEE, 2008.

## Index

```
K lân cận – K-nearest neighbor, 118
K-means clustering – phân cụm K-means, 128
  centroid – tâm cụm, 128
K-nearest neighbor – K lân cận, 118
α-sublevel set – tập dưới mức α, 315
activation function – hàm kích hoạt, 180, 218
  ReLU, 219
  sigmoid, 187, 218
  tanh, 187, 218
affine function – hàm affine, 312
argmin, 87
backpropagation – lan truyền ngược, 220
bag of words – túi từ, 92
  từ điển, 92
bao lồi – convex hull, 308
basic – cơ cở, 31
basic – cơ sở
  orthogonal – trực giao, 33
  orthonormal – trực chuẩn, 33
batch gradient descent, 171
Bayes' rule – quy tắc Bayes, 59
between-class variance – phương sai liên lớp, 290
between-class variance matrix – ma trận phương
     sai liên lớp, 292
bias – hệ số điều chỉnh, 103
bias trick – thủ thuật gộp hệ số điều chỉnh, 103,
     389
binary classification – phân loại nhị phân, 175
biến lỏng lẻo – slack variable, 326, 362
biến ngẫu nhiên – random variable, 54
biến ngẫu nhiên độc lập – independent random
     variables, 59
biến tối ưu – optimization variable, 302
biến đối ngẫu – dual variable, 339
biểu diễn one-hot – one-hot encoding, 63
biệt thức – discriminant, 289
biệt thức tuyến tính Fisher – Fisher's linear
     discriminant, 293
bài toán chính – dual problem, 339
bài toán tối ưu – convex optimization, 302
bài toán tối ưu – optimization problem, 324
                                                    bài toán tối ưu không ràng buộc – unconstrained
                                                          optimization problem, 302
                                                    bài toán tối ưu lồi – convex optimization problem,
                                                          326
                                                    bài toán đối ngẫu Lagrange – Lagrange dual
                                                          problem, 342
                                                    bất phương trình ràng buộc – inequality
                                                          constraint, 302
                                                    bầu chọn đa số – major voting, 124
                                                    bộ phân loại lề rộng nhất – maximum margin
                                                          classifier, 351
                                                    bộ phân loại naive Bayes – naive Bayes classifier,
                                                          145
                                                    chain rule – quy tắc chuỗi, 46
                                                    characteristic polynomial – đa thức đặc trưng, 35
                                                    Cholesky decomposition – Phân tích Cholesky, 39
                                                    chuyển khoảng giá trị – rescaling, 99
                                                    chuyển vị – transpose, 24
                                                    chuyển vị liên hợp – conjugate transpose, 25
                                                    chuẩn – norm, 39
                                                       chuẩn `1, 41
                                                       chuẩn `2 – `2 norm, 40
                                                       chuẩn `p, 40
                                                       chuẩn Euclid – Euclidean norm, 40
                                                       chuẩn Frobenius – Frobenius norm, 41
                                                    chuẩn hoá theo phân phối chuẩn – standardiza-
                                                          tion, 99
                                                    chéo hoá ma trận – matrix diagonalization, 37
                                                    chưa khớp – underfitting, 109
                                                    chặn dưới – lower bound, 304
                                                    chặn dưới lớn nhất – infimum, 304
                                                    chặn trên – upper bound, 304
                                                    chặn trên nhỏ nhất – supremum, 304
                                                    class boundary, 175
                                                    classification – Phân loại, 82
                                                    cluster – cụm, 128
                                                    clustering – phân cụm, 83
                                                    compact SVD – SVD giản lược, 269
                                                    complementary slackness – điều kiện lỏng lẻo bù
                                                          trừ, 344, 356
                                                    concave function – hàm lõm, 310
```

```
conditional probability – xác suất có điều kiện,
     58
conjugate distribution – phân phối liên hợp, 74
conjugate prior – tiên nghiệm liên hợp, 74
conjugate transpose – chuyển vị liên hợp, 25
consine similarity – tương tự cos, 248
constraint – ràng buộc, 302
constraint qualification – tiêu chuẩn ràng buộc,
     343
convex – lồi, 302
convex combination – tổ hợp lồi, 308
convex function – hàm lồi, 309
convex hull – bao lồi, 308
convex optimization – bài toán tối ưu, 302
convex optimization problem – bài toán tối ưu
     lồi, 326
convex set – tập lồi, 304
cross entropy – entropy chéo, 205, 319
CVXOPT, 328
căn bậc hai sai số trung bình bình phương – root
     mean squared error, 243
cơ chế kiểm soát – regularization, 113, 392
  kiểm soát `1 – `1 regularization, 114
  kiểm soát `2 – `2 regularization, 114
cơ cở – basic, 31
cơ sở – basic
  trực chuẩn – orthonormal, 33
  trực giao – orthogonal, 33
cơ sở dữ liệu khuôn mặt Yale – Yale face database,
     284
cầu chuẩn – norm ball, 306
cụm – cluster, 128
cực tiểu toàn cục – global minima, 158
cực tiểu địa phương – local minima, 158
cực trị toàn cục – global extrema, 158
cực trị địa phương – local extrema, 158
cực đại toàn cục – global maxima, 158
cực đại địa phương – local maxima, 158
đa thức – posynomial, 334
đa thức đặc trưng – characteristic polynomial, 35
đặc trưng – feature, 81
đặc trưng thủ công – hand-crafted feature, 96
đặc trưng đã trích xuất – extracted feature, 91
data point – điểm dữ liệu, 81
đầu ra dự đoán – predicted output, 100
đầu ra thực sự – ground truth, 100
determinant – định thức, 29
điều kiện bậc hai – second-order condition, 318
điều kiện bậc nhất – first-order condition, 317
điều kiện KKT – KKT condition, 345
điều kiện lỏng lẻo bù trừ – complementary
     slackness, 344, 356
điều kiện Mercer, 382
điểm dữ liệu – data point, 81
điểm khả thi – feasible point, 302, 303
điểm tối ưu – optimal point, 325
điểm tối ưu địa phương – local optimal point, 325
                                                     điểm tối ưu đối ngẫu – dual optimal point, 342
                                                     dimensionality reduction – giảm chiều dữ liệu,
                                                          92, 265
                                                     định lý siêu phẳng phân chia – separating
                                                          hyperplane theorem, 308
                                                     discriminant – biệt thức, 289
                                                     độ lệch chuẩn – standard deviation, 60, 290
                                                     độc lập tuyến tính – linearly independent, 30
                                                     đối ngẫu – duality, 338
                                                     đối ngẫu mạnh – strong duality, 343
                                                     đối ngẫu yếu – weak duality, 343
                                                     domain – tập xác định, 302
                                                     đơn thức – monomial, 334
                                                     dual feasible set – tập khả thi đối ngẫu, 342
                                                     dual optimal point – điểm tối ưu đối ngẫu, 342
                                                     dual problem – bài toán chính, 339
                                                     dual variable – biến đối ngẫu, 339
                                                     duality – đối ngẫu, 338
                                                     đường đồng mức – level sets, 166, 313
                                                     dạng toàn phương – quadratic form, 312
                                                     early stopping – kết thúc sớm, 113
                                                     eigen decomposition – phân tích riêng, 266
                                                     eigendecomposition – phân tích trị riêng, 37
                                                     eigenface – khuôn mặt riêng, 283
                                                     eigenspace – không gian riêng, 36
                                                     eigenvalues – trị riêng, 35
                                                     eigenvectors – vector riêng, 35
                                                     end-to-end, 91
                                                     entropy chéo – cross entropy, 205, 319
                                                     epoch, 172
                                                     equality constraint – phương trình ràng buộc, 302
                                                     equality constraint function – hàm phương trình
                                                          ràng buộc, 302
                                                     expectation – kỳ vọng, 59
                                                     extracted feature – đặc trưng đã trích xuất, 91
                                                     feasible point – điểm khả thi, 302, 303
                                                     feasible set – tập khả thi, 302, 303, 322
                                                     feature – đặc trưng, 81
                                                     feature extraction – trích chọn đặc trưng, 88, 265
                                                     feature selection – lựa chọn đặc trưng, 92, 114,
                                                          265
                                                     feature vector – vector đặc trưng, 81, 88
                                                     feedforward – lan truyền thuận, 217
                                                     first-order condition – điều kiện bậc nhất, 317
                                                     Fisher's linear discriminant – biệt thức tuyến
                                                          tính Fisher, 293
                                                     Gaussian naive Bayes, 147
                                                     Gaussion mixture model, 142
                                                     GD, 158
                                                     geometric programming – quy hoạch hình học,
                                                          334
                                                     giá trị suy biến – singular value, 267
                                                     giá trị tối ưu – optimal value, 325
                                                     giả chuẩn – pseudo norm, 307
                                                     giả nghịch đảo – pseudo inverse, 102
```

| giảm chiều dữ liệu – dimensionality reduction,             | sigmoid, 187, 218                                                           |  |  |
|------------------------------------------------------------|-----------------------------------------------------------------------------|--|--|
| 92, 265                                                    | tanh, 187, 218                                                              |  |  |
| global extrema – cực trị toàn cục, 158                     | hàm lõm – concave function, 310                                             |  |  |
| global maxima – cực đại toàn cục, 158                      | hàm lõm chặt – stricly concave function, 310                                |  |  |
| global minima – cực tiểu toàn cục, 158                     | hàm lồi – convex function, 309                                              |  |  |
| gradient, 43                                               | hàm lồi chặt – stricly convex function, 310                                 |  |  |
| first-order gradient – gradient bậc nhất, 43               | hàm mất mát – loss function/cost function, 86                               |  |  |
| gradient bậc hai – second-order gradient, 43               | hàm mất mát được kiểm soát – regularized loss                               |  |  |
| gradient bậc nhất – first-order gradient, 43               | function, 114                                                               |  |  |
| gradient xấp xỉ – numerical gradient, 49, 393              | hàm mật độ xác suất – probability density                                   |  |  |
| numerical gradient – gradient xấp xỉ, 49, 393              | function, 55                                                                |  |  |
| second-order gradient – gradient bậc hai, 43               | hàm phương trình ràng buộc – equality constraint                            |  |  |
| gradient descent, 158                                      | function, 302                                                               |  |  |
| batch size – kích thước batch, 172                         | hàm softmax, 202                                                            |  |  |
| kích thước batch – batch size, 172                         | hàm số Lagrange – Lagrangian, 339                                           |  |  |
| mini-batch, 172                                            | hàm trả về vector – vector-valued function, 45                              |  |  |
| momentum, 167                                              | hàm đo độ tương tự – similarity function, 246                               |  |  |
| Nesterov accelerated gradient, 170                         | hàm đối ngẫu Lagrange – the Lagrange dual                                   |  |  |
| stopping criteria – điều kiện dừng, 173                    | function, 339                                                               |  |  |
| điều kiện dừng – stopping criteria, 173                    | hạng – rank, 32                                                             |  |  |
| gradient desenct                                           | hệ số điều chỉnh – bias, 103                                                |  |  |
| stochastic gradient descent, 171                           | hệ thống gợi ý – recommendation system, 233,                                |  |  |
| grid search – tìm trên lưới, 398                           | 234                                                                         |  |  |
| ground truth – đầu ra thực sự, 100                         | dựa trên nội dung – content-based, 234                                      |  |  |
|                                                            | hiện tượng đuôi dài – long tail, 234                                        |  |  |
| Hadamard product – phép nhân từng thành                    | lọc cộng tác – collaborative filtering, 235                                 |  |  |
| phần, 26                                                   | lọc cộng tác lân cận – neighborhood-based                                   |  |  |
| Hadamard product – tích từng thành phân, 223               | collaborative filtering, 245                                                |  |  |
| halfspace – nửa không gian, 306                            | lọc cộng tác người dùng – user-user collabora                               |  |  |
| hard threshold – ngưỡng cứng, 186                          | tive filtering, 246                                                         |  |  |
| hard-margin SVM – SVM lề cứng, 362                         | lọc cộng tác sản phẩm – item-item collaborative                             |  |  |
| Hermitian, 25                                              | filtering, 251                                                              |  |  |
| Hesse – Hessian, 43, 318                                   | ma trận tiện ích – utility matrix, 235                                      |  |  |
| Hessian – Hesse, 43, 318                                   | ma trận tiện ích chuẩn hoá – normalized utility                             |  |  |
| hidden layer – tầng ẩn, 182                                | matrix, 248                                                                 |  |  |
| hierarchical classification – phân loại phân tầng,         | ma trận tương tự – similarity matrix, 248                                   |  |  |
| 197                                                        | người dùng, 234                                                             |  |  |
| hierarchical clustering – phân cụm theo tầng, 138          | sản phẩm, 234                                                               |  |  |
| hinge loss – mất mát bản lề, 369                           | học bán giám sát – semi-supervised learning, 85                             |  |  |
| hoàn thiện dữ liệu, 83                                     | học chuyển tiếp – transfer learning, 97                                     |  |  |
| hoàn thiện ma trận – matrix completion, 236                | học có giám sát – supervised learning, 84                                   |  |  |
| Huber regression – hồi quy Huber, 106                      | học củng cố – reinforcement learning, 85                                    |  |  |
| hyperparameter – siêu tham số, 75                          | học không giám sát – unsupervised learning, 84                              |  |  |
| hyperplane – siêu mặt phẳng, 305                           | học ngoại tuyến – offline learning, 81                                      |  |  |
| hyperplane – siêu phẳng, 175                               | học trực tuyến – online learning, 81, 172                                   |  |  |
| hàm affine – affine function, 312                          | hồi quy – regression, 82                                                    |  |  |
| hàm bất phương trình ràng buộc – inequality                | hồi quy Huber – Huber regression, 106                                       |  |  |
| constraint function, 302                                   | hồi quy lasso – lasso regression, 114                                       |  |  |
| hàm cơ sở radial – radial basic function, RBF,             | hồi quy logistic – logistic regression, 185                                 |  |  |
| 383                                                        | hồi quy logistic multinomial, 213                                           |  |  |
| hàm hạt nhân – kernel function, 379, 382                   | hồi quy ridge – ridge regression, 107, 114, 239                             |  |  |
| RBF, 383                                                   | hồi quy softmax – softmax regression, 201                                   |  |  |
| sigmoid, 383                                               | hồi quy tuyến tính – linear regression, 100                                 |  |  |
| tuyến tính – linear, 382                                   | hồi quy đa thức – polynomial regression, 106, 109                           |  |  |
| đa thức – polynomial, 383                                  |                                                                             |  |  |
| hàm hợp lý – likelihood, 68                                | identity matrix - ma trận đơn vị, 27                                        |  |  |
| hàm kích hoạt – activation function, 180, 218<br>ReLU, 219 | incremental matrix factorization – phân tích ma<br>trận điều chỉnh nhỏ, 264 |  |  |

independent random variables – biến ngẫu nhiên độc lập, 59 inequality constraint – bất phương trình ràng buộc, 302 inequality constraint function – hàm bất phương trình ràng buộc, 302 infimum – chặn dưới lớn nhất, 304 inner product – tích vô hướng, 26 input layer – tầng đầu vào, 180 inverse matrix - ma trận nghịch đảo, 27 iteration – vòng lặp, 172 joint probability – xác suất đồng thời, 55 kernel function – hàm hạt nhân, 379, 382 linear – tuyến tính, 382 polynomial – đa thức, 383 RBF, 383 sigmoid, 383 kernel model – mô hình hạt nhân, 378 kernel SVM – SVM hạt nhân, 378 kernel trick – thủ thuật hạt nhân, 381 khuôn mặt riêng – eigenface, 283 không gian null – null space, 31 không gian range – range space, 31 không gian riêng – eigenspace, 36 không gian sinh – span space, 30 KKT condition – điều kiện KKT, 345 KNN, 118 kết thúc sớm – early stopping, 113 kỳ vọng – expectation, 59 Lagrange dual problem – bài toán đối ngẫu Lagrange, 342 Lagrange multiplier – nhân tử Lagrange, 338 Lagrangian – hàm số Lagrange, 339 lan truyền ngược – backpropagation, 220 lan truyền thuận – feedforward, 217 Laplace smoothing – làm mềm Laplace, 147 large-scale – quy mô lớn, 119 lasso regression – hồi quy lasso, 114 layer – tầng, 217 LDA, 288 LDA đa lớp – multi-class LDA, 293 leading principal submatrix – ma trận con chính trước, 39 learning rate – tốc độ học, 160 learning rate decay – suy giảm tốc độ học, 163 left-singular value – vector suy biến trái, 267 level sets – đường đồng mức, 166, 313 likelihood – hàm hợp lý, 68 linear combination – tổ hợp tuyến tính, 30 linear constraint – ràng buộc tuyến tính, 321 linear discriminant analysis – phân tích biệt thức tuyến tính, 288 linear programming – quy hoạch tuyến tính, 329 general form – dạng tổng quát, 329 standard form – dạng tiêu chuẩn, 329 linear regression – hồi quy tuyến tính, 100 linearly dependent – phụ thuộc tuyến tính, 30 linearly independent – độc lập tuyến tính, 30 linearly separable – tách biệt tuyến tính, 175, 299, 308 local extrema – cực trị địa phương, 158 local maxima – cực đại địa phương, 158 local minima – cực tiểu địa phương, 158 local optimal point – điểm tối ưu địa phương, 325 log-likelihood, 68 logistic regression – hồi quy logistic, 185 loss function/cost function – hàm mất mát, 86 low-rank approximation – xấp xỉ hạng thấp, 271 lower bound – chặn dưới, 304 làm mềm Laplace – Laplace smoothing, 147 lồi – convex, 302 lựa chọn đặc trưng – feature selection, 92, 114, 265 ma trận chiếu – projection matrix, 92, 289 ma trận con chính – principal submatrix, 39 ma trận con chính trước – leading principal submatrix, 39 ma trận phương sai liên lớp – between-class variance matrix, 292 ma trận phương sai nội lớp – within-class variance matrix, 292 ma trận tam giác, 28 ma trận tam giác dưới, 28 ma trận tam giác trên, 28 ma trận trọng số – weight matrix, 199, 201 ma trận trực giao – orthogonal matrix, 33 ma trận unitary, 33 ma trận đường chéo, 28 ma trận đối xứng – symmetric matrix, 25 machine translation – máy dịch, 83 major voting – bầu chọn đa số, 124 MAP, 73 MAP estimation, 67 marginal probability – xác suất biên, 57 marginalization – phép biên hóa, 57 marginalization – xác suất biên, 57 matrix completion – hoàn thiện ma trận, 236 matrix diagonalization – chéo hoá ma trận, 37 maximum a posteriori estimation – ước lượng hậu nghiệm cực đại, 67 maximum a posteriori estimation, MAP estimation – ước lượng hậu nghiệm cực đại, 73 maximum likelihood estimation – ước lượng hợp lý cực đại, 67 maximum margin classifier – bộ phân loại lề rộng nhất, 351 mean squared error, MSE – sai số trung bình bình phương, 110 misclassified – phân loại lỗi, 177 MLE, 67 MNIST, 136

```
model hyperparameter – siêu tham số mô hình,
     111
model parameter – tham số mô hình, 67, 86, 111
monomial – đơn thức, 334
MSE – sai số trung bình bình phương, 221
multi-class classification – phân loại đa lớp, 196
multi-class LDA – LDA đa lớp, 293
multinomial naive Bayes, 147
máy dịch – machine translation, 83
máy vector hỗ trợ – support vector machine, 350
  lề – margin, 351
máy vector hỗ trợ đa lớp, 387
mã hoá one-hot – one-hot coding, 129
mô hình hạt nhân – kernel model, 378
mô hình thưa – sparse model, 356
mạng neuron – neural network, 180
mất mát bản lề – hinge loss, 369
mất mát bản lề tổng quát, 390
mất mát không-một – zero-one loss, 369
naive Bayes classifier – bộ phân loại naive Bayes,
     145
NBC, 145
negative definite – xác định âm, 38
negative semidefinite – nửa xác định âm, 38
neural network – mạng neuron, 180
ngưỡng – threshold, 186
ngưỡng cứng – hard threshold, 186
nhân tử Lagrange – Lagrange multiplier, 338
NMF, 264
node, unit – nút, 217
nonconvex set – tập không lồi, 305
nonnegative matrix factorization, NMF – phân
     tích ma trận không âm, 264
norm – chuẩn, 39
  `2 norm – chuẩn `2, 40
  chuẩn `1, 41
  chuẩn `p, 40
  Euclidean norm – chuẩn Euclid, 40
  Frobenius norm – chuẩn Frobenius, 41
norm ball – cầu chuẩn, 306
null space – không gian null, 31
numpy, 18
nút – node, unit, 217
nửa không gian – halfspace, 306
nửa xác định dương – positive semidefinite, 37
nửa xác định âm – negative semidefinite, 38
offline learning – học ngoại tuyến, 81
one-hot, 205
one-hot coding – mã hoá one-hot, 129
one-hot encoding – biểu diễn one-hot, 63
one-vs-one, 196
one-vs-rest, 198
online learning – học trực tuyến, 81, 172
optimal point – điểm tối ưu, 325
optimal value – giá trị tối ưu, 325
optimization problem – bài toán tối ưu, 324
                                                    optimization variable – biến tối ưu, 302
                                                    orthogonal – trực giao, 26
                                                    orthogonal matrix – ma trận trực giao, 33
                                                    output layer – tầng đầu ra, 182
                                                    overfitting – quá khớp, 108
                                                    parameter estimation – ước lượng tham số, 67
                                                    partial derivative – đạo hàm riêng, 43
                                                    patch, 94
                                                    PCA, 274
                                                    pdf, 55
                                                    perceptron learning algorithm – thuật toán học
                                                          perceptron, 175
                                                    phân cụm K-means – K-means clustering, 128
                                                      tâm cụm – centroid, 128
                                                    phân cụm – clustering, 83
                                                    phân cụm spectral – spectral clustering, 142
                                                    phân cụm theo tầng – hierarchical clustering, 138
                                                    Phân loại – classification, 82
                                                    phân loại lỗi – misclassified, 177
                                                    phân loại nhị phân – binary classification, 175
                                                    phân loại phân tầng – hierarchical classification,
                                                          197
                                                    phân loại đa lớp – multi-class classification, 196
                                                    phân phối liên hợp – conjugate distribution, 74
                                                    phân phối xác suất – probability distribution, 54,
                                                          62
                                                      phân phói Bernoulli, 62
                                                      phân phối Beta, 64
                                                      phân phối categorical, 62
                                                      phân phối chuẩn một chiều – univariate normal
                                                          distribution, 63
                                                      phân phối chuẩn nhiều chiều – multivariate
                                                          normal distribution, 63
                                                      phân phối Dirichlet, 66
                                                    phân tích biệt thức tuyến tính – linear
                                                          discriminant analysis, 288
                                                    Phân tích Cholesky – Cholesky decomposition,
                                                          39
                                                    phân tích giá trị suy biến – singular value
                                                          decomposition, 266
                                                    phân tích ma trận không âm – nonnegative
                                                          matrix factorization, NMF, 264
                                                    phân tích ma trận điều chỉnh nhỏ – incremental
                                                          matrix factorization, 264
                                                    phân tích riêng – eigen decomposition, 266
                                                    phân tích thành phần chính – principal
                                                          component analysis, 92
                                                    phân tích thành phần chính – principle
                                                          component analysis, 274
                                                    phân tích trị riêng – eigendecomposition, 37
                                                    phép biên hóa – marginalization, 57
                                                    phép nhân từng thành phần – Hadamard product,
                                                          26
                                                    phép thế ngược, 28
                                                    phép thế xuôi, 28
                                                    phương pháp elbow, 141
                                                    phương pháp nhân tử Lagrange, 402
```

| phương sai – variance, 60                                               | random variable – biến ngẫu nhiên, 54                                               |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| phương sai liên lớp – between-class variance, 290                       | range space – không gian range, 31                                                  |
| phương sai nội lớp – within-class variance, 290                         | rank – hạng, 32                                                                     |
| phương trình ràng buộc – equality constraint, 302<br>phần bù đại số, 29 | recommendation system – hệ thống gợi ý, 233,<br>234                                 |
| phổ của ma trận – spectrum, 35                                          | collaborative filtering – lọc cộng tác, 235                                         |
| phụ thuộc tuyến tính – linearly dependent, 30                           | content-based – dựa trên nội dung, 234                                              |
| PLA, 175                                                                | item-item collaborative filtering – lọc cộng tác                                    |
| pocket algorithm – thuật toán bỏ túi, 183                               | sản phẩm, 251                                                                       |
| polyhedra – siêu đa diện, 307                                           | long tail – hiện tượng đuôi dài, 234                                                |
| polynomial regression – hồi quy đa thức, 106, 109                       | neighborhood-based collaborative filtering –                                        |
| positive definite – xác định dương, 37                                  | lọc cộng tác lân cận, 245                                                           |
| positive semidefinite – nửa xác định dương, 37                          | người dùng, 234                                                                     |
| posterior probability – xác suất hậu nghiệm, 73                         | normalized utility matrix – ma trận tiện ích                                        |
| posynomial – đa thức, 334                                               | chuẩn hoá, 248                                                                      |
| predicted output – đầu ra dự đoán, 100                                  | similarity matrix – ma trận tương tự, 248                                           |
| principal component analysis – phân tích thành                          | sản phẩm, 234                                                                       |
| phần chính, 92                                                          | user-user collaborative filtering – lọc cộng tác                                    |
| principal submatrix – ma trận con chính, 39                             | người dùng, 246                                                                     |
| principle component analysis – phân tích thành<br>phần chính, 274       | utility matrix – ma trận tiện ích, 235<br>regression – hồi quy, 82                  |
| prior – tiên nghiệm, 74                                                 | regularization – cơ chế kiểm soát, 113, 392                                         |
| probability density function – hàm mật độ xác                           | `1<br>regularization – kiểm soát `1, 114                                            |
| suất, 55                                                                | `2<br>regularization – kiểm soát `2, 114                                            |
| probability distribution – phân phối xác suất, 54,<br>62                | regularization parameter – tham số kiểm soát,<br>114                                |
| multivariate normal distribution – phân phối                            | regularized loss function – hàm mất mát được                                        |
| chuẩn nhiều chiều, 63                                                   | kiểm soát, 114                                                                      |
| phân phói Bernoulli, 62                                                 | reinforcement learning – học củng cố, 85                                            |
| phân phối Beta, 64                                                      | rescaling – chuyển khoảng giá trị, 99                                               |
| phân phối categorical, 62                                               | ridge regression – hồi quy ridge, 107, 114, 239                                     |
| phân phối Dirichlet, 66                                                 | right-singular value – vector suy biến phải, 267                                    |
| univariate normal distribution – phân phối<br>chuẩn một chiều, 63       | RMSE, 243<br>root mean squared error – căn bậc hai sai số                           |
| product rule – quy tắc tích, 46                                         | trung bình bình phương, 243                                                         |
| projection matrix – ma trận chiếu, 92, 289                              | ràng buộc – constraint, 302                                                         |
| pseudo inverse – giả nghịch đảo, 102                                    | ràng buộc tuyến tính – linear constraint, 321                                       |
| pseudo norm – giả chuẩn, 307                                            |                                                                                     |
|                                                                         | sai số huấn luyện, 110                                                              |
| quadratic form – dạng toàn phương, 312                                  | sai số mô hình, 100                                                                 |
| quadratic programming – quy hoạch toàn                                  | sai số trung bình bình phương – mean squared                                        |
| phương, 331                                                             | error, MSE, 110                                                                     |
| quasiconvex – tựa lồi, 317                                              | sai số trung bình bình phương – MSE, 221                                            |
| quy hoạch hình học – geometric programming,<br>334                      | scikit-learn, 18<br>score vector – vector điểm số, 387                              |
| quy hoạch toàn phương – quadratic programming,                          | second-order condition – điều kiện bậc hai, 318                                     |
| 331                                                                     | semi-supervised learning – học bán giám sát, 85                                     |
| quy hoạch tuyến tính – linear programming, 329                          | separating hyperplane theorem – định lý siêu                                        |
| dạng tiêu chuẩn – standard form, 329                                    | phẳng phân chia, 308                                                                |
| dạng tổng quát – general form, 329                                      | SGD, 171                                                                            |
| quy mô lớn – large-scale, 119                                           | similarity function – hàm đo độ tương tự, 246                                       |
| quy tắc Bayes – Bayes' rule, 59                                         | singular value – giá trị suy biến, 267                                              |
| quy tắc chuỗi – chain rule, 46                                          | singular value decomposition – phân tích giá trị                                    |
| quy tắc tích – product rule, 46                                         | suy biến, 266                                                                       |
| quá khớp – overfitting, 108                                             | siêu mặt phẳng – hyperplane, 305                                                    |
|                                                                         | siêu phẳng – hyperplane, 175                                                        |
| radial basic function, RBF – hàm cơ sở radial,<br>383                   | siêu phẳng hỗ trợ – supporting hyperplane, 328<br>siêu tham số – hyperparameter, 75 |
|                                                                         |                                                                                     |

```
siêu tham số mô hình – model hyperparameter,
     111
siêu đa diện – polyhedra, 307
sklearn, 18
slack variable – biến lỏng lẻo, 326, 362
Slater's constraint qualification – tiêu chuẩn ràng
     buộc Slater, 343
soft-margin SVM – SVM lề mềm, 361, 362
softmax regression – hồi quy softmax, 201
span space – không gian sinh, 30
sparse model – mô hình thưa, 356
sparse vector – vector thưa, 93, 356
spectral clustering – phân cụm spectral, 142
spectrum – phổ của ma trận, 35
standard deviation – độ lệch chuẩn, 60, 290
standardization – chuẩn hoá theo phân phối
     chuẩn, 99
stricly concave function – hàm lõm chặt, 310
stricly convex function – hàm lồi chặt, 310
strong duality – đối ngẫu mạnh, 343
supervised learning – học có giám sát, 84
support vector machine – máy vector hỗ trợ, 350
  margin – lề, 351
supporting hyperplane – siêu phẳng hỗ trợ, 328
supremum, 311
supremum – chặn trên nhỏ nhất, 304
suy giảm trọng số – weight decay, 115, 190, 208,
     230, 369, 392
suy giảm tốc độ học – learning rate decay, 163
SVD, 267
SVD cắt ngọn – truncated SVD, 269
SVD giản lược – compact SVD, 269
SVM hạt nhân – kernel SVM, 378
SVM lề cứng – hard-margin SVM, 362
SVM lề mềm – soft-margin SVM, 361, 362
symmetric matrix – ma trận đối xứng, 25
tensor, 81
test set – tập kiểm tra, 81
tham số kiểm soát – regularization parameter,
     114
tham số mô hình – model parameter, 67, 86, 111
the Lagrange dual function – hàm đối ngẫu
     Lagrange, 339
threshold – ngưỡng, 186
thuật toán bỏ túi – pocket algorithm, 183
thuật toán học perceptron – perceptron learning
     algorithm, 175
thủ thuật gộp hệ số điều chỉnh – bias trick, 103,
     389
thủ thuật hạt nhân – kernel trick, 381
tinh chỉnh – fine-tuning, 97
tiên nghiệm – prior, 74
tiên nghiệm liên hợp – conjugate prior, 74
tiêu chuẩn ràng buộc – constraint qualification,
     343
tiêu chuẩn ràng buộc Slater – Slater's constraint
     qualification, 343
                                                    trace – vết, 42
                                                    training set – tập huấn luyện, 81
                                                    transpose – chuyển vị, 24
                                                    truncated SVD – SVD cắt ngọn, 269
                                                    trích chọn đặc trưng – feature extraction, 88, 265
                                                    trị riêng – eigenvalues, 35
                                                    trực giao – orthogonal, 26
                                                    tách biệt tuyến tính – linearly separable, 175,
                                                          299, 308
                                                    tìm trên lưới – grid search, 398
                                                    tích từng thành phân – Hadamard product, 223
                                                    tích vô hướng – inner product, 26
                                                    túi từ – bag of words, 92
                                                       từ điển, 92
                                                    tương tự cos – consine similarity, 248
                                                    tầng – layer, 217
                                                    tầng đầu ra – output layer, 182
                                                    tầng đầu vào – input layer, 180
                                                    tầng ẩn – hidden layer, 182
                                                    tập dưới mức α – α-sublevel set, 315
                                                    tập huấn luyện – training set, 81
                                                    tập không lồi – nonconvex set, 305
                                                    tập khả thi – feasible set, 302, 303, 322
                                                    tập khả thi đối ngẫu – dual feasible set, 342
                                                    tập kiểm tra – test set, 81
                                                    tập lồi – convex set, 304
                                                    tập xác thực – validation set, 81, 111
                                                    tập xác định – domain, 302
                                                    tốc độ học – learning rate, 160
                                                    tổ hợp lồi – convex combination, 308
                                                    tổ hợp tuyến tính – linear combination, 30
                                                    tựa lồi – quasiconvex, 317
                                                    unconstrained optimization problem – bài toán
                                                          tối ưu không ràng buộc, 302
                                                    underfitting – chưa khớp, 109
                                                    unsupervised learning – học không giám sát, 84
                                                    ước lượng hậu nghiệm cực đại – maximum a
                                                          posteriori estimation, 67
                                                    ước lượng hậu nghiệm cực đại – maximum a
                                                          posteriori estimation, MAP estimation, 73
                                                    ước lượng hợp lý cực đại – maximum likelihood
                                                          estimation, 67
                                                    ước lượng tham số – parameter estimation, 67
                                                    upper bound – chặn trên, 304
                                                    validation – xác thực, 111
                                                       cross-validation – xác thực chéo, 112, 392
                                                       leave-one-out, 112
                                                       xác thực chéo k-fold, 112
                                                    validation set – tập xác thực, 81, 111
                                                    variance – phương sai, 60
                                                    vector hoá – vectorization, 91
                                                    vector hóa – vectorization, 395
                                                    vector riêng – eigenvectors, 35
                                                    vector suy biến phải – right-singular value, 267
                                                    vector suy biến trái – left-singular value, 267
                                                    vector thưa – sparse vector, 93, 356
```

vector trọng số – weight vector, 100 vector điểm số – score vector, 387 vector đặc trưng – feature vector, 81, 88 vector-valued function – hàm trả về vector, 45 vectorization – vector hoá, 91 vectorization – vector hóa, 395 vòng lặp – iteration, 172 vết – trace, 42

weak duality – đối ngẫu yếu, 343 weight decay – suy giảm trọng số, 115, 190, 208, 230, 369, 392 weight matrix – ma trận trọng số, 199, 201 weight vector – vector trọng số, 100 within-class variance – phương sai nội lớp, 290 within-class variance matrix – ma trận phương sai nội lớp, 292

xác suất biên – marginal probability, 57

xác suất biên – marginalization, 57 xác suất có điều kiện – conditional probability, 58 xác suất hậu nghiệm – posterior probability, 73 xác suất đồng thời – joint probability, 55 xác thực – validation, 111 leave-one-out, 112 xác thực chéo – cross-validation, 112, 392 xác thực chéo k-fold, 112 xác định dương – positive definite, 37 xác định âm – negative definite, 38 xấp xỉ hạng thấp – low-rank approximation, 271

Yale face database – cơ sở dữ liệu khuôn mặt Yale, 284

zero-one loss – mất mát không-một, 369

đạo hàm riêng – partial derivative, 43 định thức – determinant, 29