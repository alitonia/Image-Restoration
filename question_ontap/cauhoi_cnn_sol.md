# Giải Đáp Câu Hỏi CNN

## 1. Câu nền tảng

**CNN là gì?**
Mạng nơ-ron tích chập (Convolutional Neural Network) là loại mạng nơ-ron chuyên dùng cho dữ liệu có cấu trúc lưới (ảnh, âm thanh), dùng phép tích chập để tự động học các đặc trưng cục bộ từ dữ liệu.

**Vì sao CNN hợp với ảnh hơn MLP thường?**
Ảnh có tính cục bộ (pixel gần nhau liên quan hơn pixel xa) và tính bất biến tịnh tiến (vật thể ở góc nào cũng nên nhận ra được). CNN khai thác hai tính chất này, còn MLP xử lý mọi vị trí như nhau, dẫn tới quá nhiều tham số và không hiệu quả.

**CNN tận dụng tính chất nào: local connectivity hay weight sharing?**
Cả hai. Local connectivity giúp mỗi neuron chỉ "nhìn" một vùng nhỏ của input. Weight sharing giúp cùng một bộ lọc được dùng ở mọi vị trí, giảm số tham số.

**Weight sharing là gì?**
Một filter dùng cùng bộ tham số để quét toàn bộ ảnh, thay vì mỗi vị trí có tham số riêng. Điều này giảm mạnh số tham số và giúp phát hiện cùng đặc trưng ở nhiều nơi trong ảnh.

**Local receptive field là gì?**
Vùng nhỏ của input mà mỗi neuron kết nối tới. Thay vì kết nối toàn bộ input, mỗi neuron chỉ "nhìn" một vùng hạn chế, phù hợp với tính cục bộ của ảnh.

**Feature map là gì?**
Kết quả của việc áp một filter lên ảnh. Mỗi giá trị trong feature map biểu diễn mức độ xuất hiện của một pattern cụ thể tại một vị trí.

**Kernel/filter là gì?**
Ma trận nhỏ chứa các tham số học được, dùng để trượt qua ảnh và tính tích chập nhằm phát hiện một đặc trưng cụ thể (cạnh, màu, texture...).

**Một convolution layer thật ra đang học cái gì?**
Học các bộ lọc (weights) sao cho mỗi filter kích hoạt mạnh khi gặp một pattern nhất định trong ảnh — ví dụ cạnh ngang, cạnh dọc, góc, v.v.

**Vì sao CNN ít tham số hơn fully connected network trên ảnh lớn?**
Nhờ weight sharing: một filter chỉ có k×k tham số nhưng được dùng ở mọi vị trí. Trong khi đó, fully connected yêu cầu một tham số riêng cho mỗi kết nối giữa pixel và neuron.

**Vì sao CNN có tính chất translation-related tốt hơn MLP?**
Nhờ weight sharing và pooling, CNN phát hiện đặc trưng dù nó ở vị trí nào trong ảnh, tạo ra tính bất biến tịnh tiến gần đúng.

**Input tensor trong CNN thường có dạng gì?**
`(batch_size, channels, height, width)` — ví dụ ảnh màu RGB là `(N, 3, H, W)`.

**Channel trong ảnh và channel trong feature map khác nhau thế nào?**
Channel trong ảnh là kênh màu (RGB = 3 channels). Channel trong feature map là số filter đã áp, mỗi channel biểu diễn một loại đặc trưng khác nhau học được từ layer đó.

---

## 2. Convolution, padding, stride, output size

**Convolution trong CNN làm gì về trực giác?**
Trượt một filter nhỏ qua ảnh, tại mỗi vị trí tính tích vô hướng giữa filter và vùng ảnh tương ứng, tạo ra một giá trị duy nhất. Kết quả cho thấy pattern trong filter xuất hiện mạnh hay yếu tại vị trí đó.

**Vì sao một filter có thể phát hiện cạnh, texture, hay pattern?**
Vì các filter học được sẽ phản ứng mạnh (activation lớn) khi vùng ảnh trùng khớp với pattern mà filter biểu diễn. Việc học là qua gradient descent.

**Stride là gì?**
Bước trượt của filter qua ảnh. Stride=1 thì filter di chuyển từng pixel, stride=2 thì nhảy 2 pixel mỗi lần, làm giảm kích thước output.

**Padding là gì?**
Thêm giá trị (thường là 0) xung quanh biên ảnh trước khi áp filter, để kiểm soát kích thước output và cho filter "nhìn thấy" các pixel ở biên.

**Zero-padding dùng để làm gì?**
Giữ nguyên kích thước spatial của feature map sau convolution, và cho phép filter xử lý tốt hơn các pixel ở các góc và cạnh.

**Nếu không padding thì kích thước feature map thay đổi ra sao?**
Feature map nhỏ hơn input sau mỗi layer. Với filter k×k, mỗi chiều giảm đi `k-1` pixel. Sau nhiều layer, kích thước có thể co lại rất nhanh.

**Công thức tính output size của conv layer là gì?**
`output = floor((input - kernel + 2*padding) / stride) + 1`

**Khi nào đầu ra "không fit" và vì sao?**
Khi `(input - kernel + 2*padding)` không chia hết cho stride, phép chia sẽ có phần dư, và ta thường dùng `floor` để xử lý, nhưng một số framework báo lỗi nếu không cẩn thận.

**Kernel size lớn hay nhỏ ảnh hưởng gì?**
Kernel lớn có receptive field lớn hơn nhưng nhiều tham số và tính toán hơn. Kernel nhỏ (3×3) thường hiệu quả hơn khi dùng nhiều lớp chồng lên nhau.

**Dùng nhiều kernel trong một lớp có ý nghĩa gì?**
Mỗi kernel học một đặc trưng khác nhau. Kết quả là nhiều feature map song song, tương ứng với nhiều "khía cạnh" khác nhau của ảnh được phát hiện cùng lúc.

**Receptive field là gì?**
Vùng ảnh gốc mà một neuron trong feature map nào đó "nhìn thấy" được sau khi tính ngược qua các lớp. Receptive field lớn hơn = neuron nắm bắt được thông tin từ vùng rộng hơn.

**Receptive field tăng như thế nào khi chồng nhiều lớp conv?**
Tăng dần theo số lớp. Sau `n` lớp conv với kernel `k×k` và stride 1, receptive field tăng thêm `k-1` mỗi lớp. Sau bước pool hoặc stride>1, receptive field tăng nhanh hơn.

**Vì sao nhiều lớp 3×3 có thể thay cho một lớp kernel lớn?**
Hai lớp 3×3 cho receptive field 5×5, ba lớp cho 7×7 — tương đương kernel lớn — nhưng có ít tham số hơn và thêm activation phi tuyến giữa các lớp, tăng khả năng biểu diễn.

**Dilation convolution là gì?**
Filter trượt nhưng các phần tử bên trong filter có khoảng cách được giãn ra (dilation rate > 1). Điều này giúp tăng receptive field mà không cần tăng số tham số hay giảm độ phân giải.

**Dilation giúp gì cho receptive field?**
Tăng receptive field nhanh hơn mà không cần stride hoặc pooling, giữ nguyên kích thước feature map, phù hợp cho các task như segmentation.

**Convolution và cross-correlation khác nhau về mặt toán học thế nào?**
Convolution thật sự cần flip kernel trước khi tính. Cross-correlation không flip. Trong CNN, phép tính thường là cross-correlation, nhưng vẫn gọi là "convolution" theo quy ước vì filter được học tự động nên flip hay không không quan trọng.

---

## 3. Activation và pooling

**Vì sao phải có activation sau convolution?**
Convolution là phép tuyến tính. Nếu không có activation phi tuyến, nhiều lớp tuyến tính chồng nhau vẫn chỉ tương đương một lớp tuyến tính duy nhất, làm mất khả năng biểu diễn phức tạp.

**ReLU giúp CNN tốt hơn sigmoid/tanh ở điểm nào?**
ReLU không gây vanishing gradient khi giá trị dương, tính toán nhanh, và hội tụ trong thực hành tốt hơn. Sigmoid/tanh bão hòa ở hai đầu khiến gradient vanish trong mạng sâu.

**Max pooling là gì?**
Chia feature map thành các ô nhỏ, lấy giá trị lớn nhất trong mỗi ô. Giảm kích thước spatial và giữ lại đặc trưng nổi bật nhất trong vùng.

**Average pooling là gì?**
Lấy giá trị trung bình trong mỗi ô. Giữ thông tin phân bố đều hơn trong vùng, nhưng làm mờ thông tin khu vực.

**Max pooling và average pooling khác nhau thế nào?**
Max pooling giữ đặc trưng mạnh nhất (tốt cho phát hiện sự hiện diện của pattern), average pooling giữ thông tin tổng hợp của cả vùng (tốt khi cần biểu diễn phân bố hơn là vị trí cụ thể).

**Pooling giúp giảm chi phí tính toán ra sao?**
Giảm kích thước không gian (height × width) của feature map, từ đó giảm số phép tính trong các lớp tiếp theo và giảm sử dụng bộ nhớ.

**Pooling tạo invariance theo trực giác nào?**
Bằng cách lấy giá trị đại diện trong một vùng nhỏ, mô hình ít nhạy cảm hơn với dịch chuyển nhỏ của đặc trưng trong vùng đó — tạo bất biến cục bộ.

**Vì sao pooling có thể làm mất thông tin vị trí?**
Vì pooling gộp nhiều giá trị từ một vùng thành một giá trị duy nhất, thông tin về vị trí chính xác của pattern trong vùng bị mất.

**Khi nào nên bỏ pooling và thay bằng strided convolution?**
Khi vị trí thông tin quan trọng (detection, segmentation), hoặc khi muốn mạng tự học cách giảm chiều thay vì dùng pooling cố định. Strided convolution linh hoạt và học được hơn.

**Global average pooling là gì?**
Lấy trung bình toàn bộ feature map mỗi channel thành một giá trị duy nhất. Biến feature map `(C, H, W)` thành vector `(C,)`, loại bỏ hoàn toàn thông tin vị trí.

**Vì sao global average pooling hay thay fully connected layer ở cuối mạng?**
Giảm số tham số rất nhiều, giảm overfitting, và có thể xử lý ảnh kích thước khác nhau mà không cần thay đổi kiến trúc.

**Pooling có phải lúc nào cũng tốt không?**
Không. Với các task cần vị trí chính xác (segmentation, keypoint detection), pooling làm mất thông tin quan trọng. Trong nhiều kiến trúc hiện đại, strided conv thay cho pooling.

---

## 4. Kiến trúc CNN điển hình

**Một CNN điển hình gồm những block nào?**
Phần trích xuất đặc trưng: các lớp conv → activation → pooling lặp lại. Phần phân loại: flatten → fully connected layers → softmax (hoặc global average pooling → FC). 

**Vì sao layer đầu học đặc trưng đơn giản còn layer sâu học đặc trưng phức tạp?**
Layer đầu có receptive field nhỏ, chỉ thấy vùng ảnh nhỏ → học cạnh, màu. Layer sâu hơn kết hợp thông tin từ nhiều vùng rộng hơn → học texture, part, rồi whole object.

**Tại sao depth quan trọng trong CNN?**
Mạng sâu có thể biểu diễn hàm phi tuyến phức tạp hơn với ít tham số hơn mạng nông. Mỗi lớp xây dựng trừu tượng hóa cao hơn từ lớp trước.

**Vì sao không dùng kernel rất lớn ngay từ đầu cho mọi lớp?**
Kernel lớn có nhiều tham số, tốn bộ nhớ, chậm, và dễ overfit. Chồng nhiều lớp kernel nhỏ hiệu quả hơn về tham số và thêm thêm phi tuyến giữa các lớp.

**Conv block là gì?**
Một đơn vị lặp lại trong CNN, thường gồm: Conv → BatchNorm → ReLU (→ Pooling tuỳ chọn). Nhiều conv block chồng lên nhau tạo thành backbone.

**Classifier head trong CNN thường gồm gì?**
Sau backbone: flatten hoặc global avg pooling, rồi một hoặc vài fully connected layers, và cuối cùng là lớp output với softmax cho classification.

**Fully connected layers ở cuối mạng để làm gì?**
Kết hợp toàn bộ đặc trưng đã trích xuất từ mọi vị trí để đưa ra quyết định phân loại tổng thể, không phụ thuộc vào vị trí cụ thể nữa.

**Khi nào có thể bỏ FC layers hoàn toàn?**
Khi dùng global average pooling, hoặc trong các task như detection/segmentation nơi output giữ cấu trúc không gian. Nhiều kiến trúc hiện đại không dùng FC layers.

**Vì sao CNN sâu hơn thường mạnh hơn nhưng khó train hơn?**
Sâu hơn → biểu diễn tốt hơn, nhưng gradient vanish qua nhiều lớp, và optimization landscape phức tạp hơn. ResNet giải quyết vấn đề này bằng skip connections.

**Phân biệt shallow CNN và deep CNN:**
Shallow CNN có ít lớp (khó học đặc trưng phức tạp, có thể đủ cho bài toán đơn giản). Deep CNN có nhiều lớp hơn, học được đặc trưng trừu tượng hơn nhưng cần kỹ thuật đặc biệt để train ổn định.

---

## 5. LeNet, AlexNet, VGG, Inception, ResNet

**LeNet-5 là gì?**
Một trong những CNN đầu tiên, được thiết kế bởi Yann LeCun vào cuối thập niên 1980–1990 để nhận dạng chữ số viết tay (MNIST).

**Vì sao LeNet-5 được xem là CNN kinh điển đầu tiên?**
Vì nó lần đầu kết hợp convolution, pooling, và fully connected layers thành một kiến trúc end-to-end hoàn chỉnh, chứng minh CNN hoạt động thực tiễn.

**LeNet được dùng cho bài toán gì?**
Nhận dạng chữ số viết tay, ứng dụng ban đầu là đọc số trên séc ngân hàng.

**AlexNet đột phá ở điểm nào?**
Thắng ImageNet 2012 với khoảng cách rất lớn so với các phương pháp khác, nhờ kết hợp: mạng sâu 5 conv + 3 FC, ReLU activation, dropout, data augmentation, và huấn luyện trên GPU.

**Vì sao AlexNet là mốc quan trọng của deep learning cho vision?**
Chứng minh rằng mạng nơ-ron sâu + GPU + big data có thể vượt xa các phương pháp feature engineering thủ công, khai mào kỷ nguyên deep learning hiện đại.

**AlexNet dùng ReLU có lợi gì?**
ReLU hội tụ nhanh hơn tanh/sigmoid khoảng 6 lần trong thực nghiệm của paper, vì không bị vanishing gradient khi giá trị dương.

**AlexNet dùng dropout để làm gì?**
Giảm overfitting bằng cách ngẫu nhiên vô hiệu hóa một tỷ lệ neurons trong khi train, buộc mạng học các biểu diễn thừa dự phòng.

**Vai trò của GPU trong thành công của AlexNet là gì?**
GPU song song hóa tính toán ma trận, giúp train mạng lớn trên ImageNet trong thời gian khả thi. Không có GPU, AlexNet sẽ không thực tiễn.

**VGG khác AlexNet ở điểm nào?**
VGG sâu hơn (16–19 lớp so với 8), dùng toàn bộ filter nhỏ 3×3 thay vì filter lớn 11×11 hay 5×5, và thiết kế đơn giản, nhất quán hơn.

**Vì sao VGG chuộng nhiều conv 3×3 nhỏ?**
Hai lớp 3×3 có receptive field tương đương một lớp 5×5 nhưng ít tham số hơn và có thêm activation phi tuyến, giúp mạng mạnh hơn.

**Lợi ích của việc thay kernel lớn bằng nhiều lớp 3×3 là gì?**
Ít tham số hơn, thêm phi tuyến, và nhìn chung biểu diễn tốt hơn với cùng receptive field.

**Nhược điểm lớn của VGG là gì?**
Rất nhiều tham số (138M cho VGG-16), chủ yếu do các FC layers, dẫn tới chậm và tốn bộ nhớ.

**Inception module là gì?**
Một khối trong kiến trúc Inception (GoogLeNet) gồm nhiều nhánh filter song song kích thước khác nhau (1×1, 3×3, 5×5) và max pooling 3×3, kết quả từ các nhánh được ghép lại theo chiều channel.

**Vì sao Inception dùng nhiều nhánh kernel khác nhau song song?**
Cho phép mạng chú ý đặc trưng ở nhiều kích thước không gian khác nhau trong cùng một lớp mà không cần chọn trước kích thước filter.

**1×1 convolution trong Inception để làm gì?**
Giảm chiều channel (bottleneck) trước khi áp filter 3×3 hay 5×5, giảm mạnh số phép tính mà không làm mất thông tin không gian.

**Vì sao Inception hiệu quả hơn về tham số?**
Dùng 1×1 conv để giảm chiều trước khi tính toán lớn, kết hợp các kích thước filter song song, đạt hiệu suất cao hơn VGG với ít tham số hơn nhiều.

**ResNet giải quyết vấn đề gì?**
Vấn đề "degradation": khi mạng quá sâu (>20 lớp với kiến trúc thông thường), độ chính xác trên cả train set bắt đầu giảm — không phải do overfitting mà do khó tối ưu hơn.

**"Degradation problem" là gì?**
Hiện tượng mạng sâu hơn cho kết quả tệ hơn mạng nông hơn ngay cả trên tập train, do gradient không truyền tốt qua quá nhiều lớp.

**Residual block là gì?**
Một khối gồm một hoặc vài conv layers, với một skip connection (shortcut) cộng thẳng input ban đầu vào output: `output = F(x) + x`.

**Skip connection giúp gì cho gradient và optimization?**
Tạo "đường cao tốc" cho gradient chạy thẳng từ output về input mà không qua phép nhân phi tuyến, giúp gradient không vanish dù mạng rất sâu.

**Vì sao học residual mapping F(x) dễ hơn học trực tiếp H(x)?**
Nếu layer cần học identity (không thay đổi gì), thì F(x)=0 dễ học hơn H(x)=x, vì đẩy weights về 0 đơn giản hơn học identity mapping chính xác.

**ResNet khác VGG về khả năng mở rộng độ sâu như thế nào?**
ResNet có thể train đến 152 lớp (hay hơn nữa) nhờ skip connections. VGG bị giới hạn ở 19 lớp vì gradient vanish khiến mạng sâu hơn không train được tốt.

---

## 6. Batch norm, regularization, và training

**Vì sao CNN dễ overfit khi dữ liệu ít?**
CNN có nhiều tham số, có thể ghi nhớ dữ liệu thay vì học pattern tổng quát. Ít dữ liệu → ít ràng buộc → mô hình dễ học "vẹt".

**Dropout trong CNN dùng để làm gì?**
Ngẫu nhiên bỏ một phần neurons trong khi train (thường áp vào FC layers), buộc mạng học nhiều con đường độc lập để dự đoán, giảm overfitting.

**Data augmentation phổ biến cho ảnh là gì?**
Flip ngang/dọc, crop ngẫu nhiên, xoay, thay đổi độ sáng/tương phản, color jitter, cutout, mixup... giúp tăng sự đa dạng dữ liệu train mà không cần thu thập thêm ảnh.

**Vì sao augmentation đặc biệt quan trọng trong computer vision?**
Ảnh thật thế giới xuất hiện ở mọi góc độ, ánh sáng, kích thước. Augmentation giúp mô hình học tính bất biến này mà không cần vô số ảnh thật.

**Batch normalization giúp gì khi train CNN?**
Chuẩn hóa activation trong mỗi mini-batch về mean=0, variance=1, giúp training ổn định hơn, cho phép learning rate lớn hơn, giảm nhu cầu khởi tạo trọng số cẩn thận, và có hiệu ứng regularization nhẹ.

**Learning rate ảnh hưởng thế nào tới CNN training?**
Quá lớn: loss dao động hoặc phân kỳ. Quá nhỏ: hội tụ rất chậm. Thường dùng learning rate schedule (giảm dần) hoặc warmup để cân bằng.

**Vì sao SGD + momentum hay dùng cho CNN?**
SGD generalize tốt hơn Adam trong nhiều benchmark vision. Momentum giúp vượt qua các vùng phẳng và dao động ít hơn, hội tụ nhanh hơn SGD thuần.

**Weight decay dùng để làm gì?**
Là L2 regularization: thêm penalty vào weights lớn trong loss, khuyến khích weights nhỏ hơn, giảm overfitting.

**Khi train loss giảm mà val loss tăng, đó là dấu hiệu gì?**
Overfitting: mô hình ghi nhớ train set nhưng không tổng quát hóa sang dữ liệu mới.

**Khi model không học được gì, nên kiểm tra gì trước?**
Learning rate (có thể quá lớn/nhỏ), khởi tạo weights, data pipeline (shuffle, normalize), loss function, và xem gradient có bằng 0 không.

**Vì sao initialization quan trọng với mạng sâu?**
Nếu weights quá lớn/nhỏ từ đầu, activation sẽ bão hòa hoặc về 0 ngay, khiến gradient vanish ngay từ bước đầu tiên. Khởi tạo tốt (Xavier, He) giữ phân phối activation ổn định.

**Vì sao gradient vanishing/exploding vẫn có thể là vấn đề trong CNN sâu?**
Dù ReLU giảm vanishing cho các lớp dương, nhưng mạng rất sâu không có skip connections vẫn có thể gặp vanishing. Exploding gradient vẫn xảy ra nếu learning rate lớn hoặc khởi tạo không cẩn thận.

**Khi nào nên dùng early stopping?**
Khi val loss bắt đầu tăng trong khi train loss vẫn giảm — dừng training ở điểm val loss tốt nhất để tránh overfitting.

**Label smoothing có lợi gì trong classification?**
Thay vì one-hot labels (0/1 cứng), dùng nhãn mềm (ví dụ 0.9/0.1). Giúp mô hình ít tự tin thái quá, cải thiện calibration và generalization.

---

## 7. Transfer learning và fine-tuning

**Transfer learning là gì trong CNN?**
Dùng lại một CNN đã được huấn luyện trên tập dữ liệu lớn (như ImageNet) làm điểm xuất phát cho bài toán mới, thay vì train từ đầu.

**Vì sao pretrained CNN hữu ích khi dữ liệu đích ít?**
Các đặc trưng học được từ ảnh tự nhiên (cạnh, texture, shape...) thường tổng quát tốt. Dùng lại chúng giúp tránh overfitting khi dữ liệu ít.

**Feature extractor và fine-tuning khác nhau thế nào?**
Feature extractor: đóng băng (freeze) toàn bộ backbone, chỉ train lớp classifier head mới. Fine-tuning: mở (unfreeze) một số hoặc toàn bộ backbone, cho phép chúng tiếp tục học trên dữ liệu mới.

**Khi nào chỉ nên freeze backbone?**
Khi dữ liệu mới rất ít và/hoặc rất giống với dữ liệu pretrain. Freeze giảm nguy cơ overfitting và giảm chi phí tính toán.

**Khi nào nên unfreeze nhiều layer hơn?**
Khi dữ liệu mới đủ nhiều hoặc domain mới khá khác với ImageNet. Unfreeze cho phép backbone thích nghi với đặc trưng của domain mới.

**Vì sao layer đầu thường transfer tốt hơn layer cuối?**
Layer đầu học đặc trưng chung như cạnh, màu — có thể dùng cho mọi ảnh. Layer sâu học đặc trưng rất cụ thể với dataset gốc (ví dụ: bộ phận chó mèo), ít tổng quát hơn.

**Domain shift ảnh hưởng transfer learning ra sao?**
Nếu domain mới rất khác (ví dụ ảnh y tế so với ImageNet), các đặc trưng học được có thể không còn phù hợp, hiệu quả transfer giảm, có thể cần fine-tune nhiều hơn hoặc từ các lớp sớm hơn.

**Nếu dữ liệu mới rất khác ImageNet, nên fine-tune thế nào?**
Unfreeze nhiều lớp hơn hoặc toàn bộ mạng, dùng learning rate nhỏ để không phá vỡ hoàn toàn knowledge đã học, cần nhiều dữ liệu hơn để tránh overfitting.

**Vì sao CNN pretrained từng là backbone mặc định cho detection/segmentation?**
Vì ImageNet pretraining cho đặc trưng thị giác tốt, có thể tái sử dụng cho các task phức tạp hơn. Hiện nay ViT-based backbones cũng đang thay thế dần.

**CNN feature backbone là gì?**
Phần trunk của CNN (thường không có FC head) được dùng để trích xuất feature maps cho các task khác như detection, segmentation, và được kết nối với detection head hoặc segmentation head.

---

## 8. Bài toán vision thường đi với CNN

**Image classification khác object detection ở đâu?**
Classification: gán nhãn cho toàn bộ ảnh. Detection: xác định vị trí (bounding box) và nhãn của từng vật thể trong ảnh.

**Detection khác segmentation ở đâu?**
Detection: cho bounding box bao quanh vật thể. Segmentation: cho biết pixel nào thuộc vật thể nào — chi tiết hơn về vị trí.

**Vì sao CNN phù hợp cho classification?**
CNN trích xuất đặc trưng có tính trừu tượng tăng dần qua các lớp, cuối cùng biểu diễn toàn bộ ảnh thành vector đặc trưng để phân loại.

**Vì sao detection cần vừa "nhìn thấy cái gì" vừa "ở đâu"?**
Phải nhận ra vật thể (classification) và đồng thời xác định vị trí chính xác (regression bounding box), hai task cùng thực hiện song song.

**Semantic segmentation và instance segmentation khác nhau thế nào?**
Semantic: gán nhãn lớp cho mỗi pixel nhưng không phân biệt hai vật thể cùng lớp. Instance: phân biệt từng cá thể riêng biệt, kể cả hai người đứng cạnh nhau.

**Feature pyramid trong vision có ý nghĩa gì?**
Kết hợp feature maps ở nhiều độ phân giải khác nhau (từ nhiều lớp CNN), giúp phát hiện vật thể ở nhiều kích thước — nhỏ lẫn lớn.

**Vì sao đặc trưng nhiều mức của CNN hữu ích cho detection/segmentation?**
Đặc trưng sớm (fine-grained, high-res) tốt cho vật thể nhỏ. Đặc trưng sâu (semantic, low-res) tốt cho nhận dạng ngữ nghĩa. Kết hợp cả hai giúp phát hiện tốt ở mọi kích thước.

**CNN có thể dùng cho video bằng cách nào?**
Xử lý từng frame độc lập (2D CNN), hoặc dùng 3D CNN để học đặc trưng theo cả không gian lẫn thời gian, hoặc kết hợp với LSTM/Transformer sau backbone CNN.

---

## 9. Hạn chế của CNN

**Hạn chế lớn của CNN so với Transformer là gì?**
CNN có inductive bias mạnh về locality — khó học quan hệ xa trong ảnh. Transformer (self-attention) trực tiếp kết nối mọi vị trí, nên học phụ thuộc xa tốt hơn, đặc biệt khi dữ liệu đủ lớn.

**Vì sao CNN thiên về local patterns?**
Vì mỗi neuron chỉ kết nối với một vùng nhỏ (receptive field ban đầu). Phải chồng nhiều lớp để receptive field mở rộng đủ để thấy toàn bộ ảnh.

**Vì sao modeling long-range dependencies khó hơn trong CNN?**
Phụ thuộc xa phải truyền qua nhiều lớp mới "gặp nhau". Trong khi đó, self-attention trong Transformer kết nối mọi vị trí trong một bước.

**CNN có thật sự bất biến tịnh tiến hoàn toàn không?**
Không hoàn toàn. CNN có equivariance (khi vật thể dịch chuyển, feature map dịch chuyển tương ứng). Pooling tạo ra invariance gần đúng, nhưng không hoàn hảo.

**Vì sao pooling vừa có lợi vừa có hại?**
Lợi: giảm kích thước, tạo invariance cục bộ. Hại: mất thông tin vị trí chính xác, có thể bất lợi cho detection/segmentation.

**Khi dữ liệu rất lớn, vì sao Transformer có thể cạnh tranh hoặc vượt CNN?**
Transformer không có inductive bias locality nên cần nhiều dữ liệu hơn để học từng đầu. Khi có đủ dữ liệu lớn, self-attention linh hoạt hơn và học được các pattern phức tạp hơn.

**CNN có inductive bias gì mạnh?**
Locality (chỉ xem vùng nhỏ), translation equivariance (weight sharing), và phân cấp đặc trưng từ đơn giản đến phức tạp theo độ sâu.

**Khi nào inductive bias của CNN là lợi thế?**
Khi dữ liệu ít hoặc vừa phải, inductive bias giúp học nhanh hơn, tổng quát tốt hơn mà không cần quá nhiều ví dụ training.

---

## 10. Phân biệt

**Convolution vs fully connected:** Conv dùng filter cục bộ với weight sharing → ít tham số, phù hợp ảnh. FC kết nối mọi input với mọi output → nhiều tham số, không khai thác locality.

**Kernel vs feature map:** Kernel là bộ lọc (tham số cần học). Feature map là kết quả đầu ra sau khi áp kernel lên input.

**Padding vs stride:** Padding thêm viền xung quanh input để điều khiển kích thước output. Stride là bước nhảy của filter, stride lớn → output nhỏ hơn.

**Max pooling vs average pooling:** Max giữ giá trị nổi bật nhất. Average giữ thông tin trung bình của cả vùng. Max thường tốt hơn cho phát hiện đặc trưng rõ ràng.

**Shallow CNN vs deep CNN:** Shallow: ít lớp, đặc trưng đơn giản, phù hợp bài toán dễ. Deep: nhiều lớp, đặc trưng phức tạp, cần kỹ thuật huấn luyện tốt hơn (residual, batch norm).

**AlexNet vs VGG:** AlexNet dùng filter lớn (11×11), ít lớp hơn. VGG dùng toàn 3×3, nhiều lớp hơn, nhất quán hơn, hiệu quả hơn về tham số/hiệu năng.

**VGG vs Inception:** VGG: đơn giản, nhiều tham số. Inception: nhiều nhánh song song nhiều kích thước filter, hiệu quả hơn về tham số và tính toán.

**VGG vs ResNet:** VGG bị giới hạn độ sâu. ResNet dùng skip connections nên có thể rất sâu (100+ lớp) mà vẫn train được.

**Residual connection vs plain stacked layers:** Plain: gradient phải qua toàn bộ phép nhân ở mỗi lớp → dễ vanish. Residual: có đường bypass cộng thẳng → gradient chạy tốt hơn.

**Feature extraction vs fine-tuning:** Feature extraction: freeze backbone, chỉ train head. Fine-tuning: mở một phần hoặc toàn bộ backbone để tiếp tục học.

---

## 11. Câu tự luận

**Cơ chế hoạt động của convolution layer trong CNN:**
Filter (kernel) trượt qua ảnh với stride định sẵn. Tại mỗi vị trí, tính tích vô hướng giữa filter và vùng ảnh tương ứng, cộng bias, rồi qua activation. Nhiều filter song song tạo ra nhiều feature maps, mỗi cái phát hiện một loại đặc trưng. Quá trình học: cập nhật filter qua backpropagation để minimize loss.

**Vai trò của padding, stride, và pooling:**
- Padding: kiểm soát kích thước output, cho phép xử lý biên ảnh tốt hơn.
- Stride: kiểm soát bước nhảy, stride lớn = output nhỏ hơn, tính toán ít hơn.
- Pooling: giảm kích thước spatial, tạo bất biến cục bộ, giảm overfitting và chi phí tính toán.

**Vì sao CNN hiệu quả hơn MLP cho dữ liệu ảnh:**
1. Weight sharing: cùng filter dùng ở mọi vị trí → ít tham số hơn rất nhiều so với FC kết nối toàn bộ.
2. Local connectivity: chỉ xem vùng lân cận → phù hợp với tính cục bộ của ảnh.
3. Phân cấp đặc trưng: tự động học từ cạnh/màu → texture → shape → object level.

**So sánh AlexNet, VGG, Inception, ResNet:**
| Mô hình | Đặc điểm chính |
|---------|---------------|
| AlexNet | 5 conv + 3 FC, filter lớn, ReLU + dropout lần đầu |
| VGG | Nhiều lớp 3×3, đơn giản, nhiều tham số |
| Inception | Nhiều nhánh kích thước khác nhau, 1×1 bottleneck, hiệu quả tham số |
| ResNet | Skip connections, train được rất sâu, giải quyết degradation |

**Vì sao ResNet giải quyết được degradation problem:**
Với plain network, khi thêm lớp, gradient phải nhân qua nhiều phép biến đổi → vanish → lớp mới không học được gì hữu ích, thậm chí làm hỏng lớp trước. ResNet thêm skip connection `F(x) + x`: gradient có thể đi thẳng qua shortcut mà không qua phép nhân, giữ được gradient đủ lớn dù mạng rất sâu. Đồng thời, mỗi layer chỉ cần học residual (phần thêm nhỏ), dễ hơn học full mapping.

**Cách dùng transfer learning với CNN khi dữ liệu ít:**
1. Lấy CNN pretrained (VGG, ResNet...) đã train trên ImageNet.
2. Freeze toàn bộ backbone (không update weights của backbone).
3. Thay lớp classifier head cuối bằng lớp mới phù hợp số class bài toán.
4. Train chỉ lớp head mới trên dữ liệu ít của mình với learning rate phù hợp.
5. Nếu dữ liệu đủ: unfreeze thêm vài lớp cuối backbone và fine-tune toàn bộ với learning rate rất nhỏ.
