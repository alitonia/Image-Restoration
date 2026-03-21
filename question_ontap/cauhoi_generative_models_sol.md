# Giải Đáp Câu Hỏi Generative Models

## 1. Câu nền tảng

**Generative model là gì?**
Mô hình học phân phối xác suất của dữ liệu `p(x)` để có thể sinh ra các mẫu mới trông giống dữ liệu thật, hoặc ước lượng khả năng một điểm dữ liệu xuất hiện.

**Generative model khác discriminative model ở đâu?**
Discriminative: học `p(y|x)` — phân biệt nhãn từ input (ví dụ classifier). Generative: học `p(x)` hoặc `p(x,y)` — mô hình hóa bản thân dữ liệu, không chỉ ranh giới phân loại.

**"Học phân phối dữ liệu" nghĩa là gì?**
Nghĩa là mô hình học được cách dữ liệu thật được tạo ra — vùng nào trong không gian input có xác suất cao (tồn tại nhiều ảnh thật ở đó), vùng nào xác suất thấp (ít hoặc không có ảnh thật).

**Sampling từ mô hình sinh là gì?**
Dùng mô hình đã học `p(x)` để tạo ra điểm dữ liệu mới — ví dụ sinh ra ảnh mới trông như ảnh thật mà không cần copy từ tập train.

**Likelihood là gì? Vì sao generative modeling hay gắn với likelihood?**
Likelihood `p(x; θ)` là xác suất mô hình gán cho điểm dữ liệu `x`. Generative modeling thường học bằng cách tối đa hóa likelihood trên dữ liệu train (maximum likelihood estimation), nghĩa là làm cho mô hình "tin" dữ liệu thật là có xác suất cao.

**Explicit density model và implicit density model khác nhau thế nào?**
Explicit: mô hình định nghĩa và có thể tính trực tiếp `p(x)` (như VAE, flows, autoregressive). Implicit: chỉ biết cách lấy mẫu từ `p(x)` nhưng không tính được giá trị `p(x)` cụ thể (như GAN).

**Latent variable model là gì?**
Mô hình giả định mỗi điểm dữ liệu `x` được tạo ra từ một biến ẩn `z` không quan sát được: `p(x) = ∫ p(x|z)p(z) dz`. Biến `z` biểu diễn các yếu tố tiềm ẩn tạo nên dữ liệu.

**Prior, posterior, likelihood trong mô hình latent có vai trò gì?**
- **Prior** `p(z)`: giả định về biến ẩn trước khi thấy dữ liệu (thường là Gaussian).
- **Likelihood** `p(x|z)`: xác suất sinh ra `x` từ `z` (decoder).
- **Posterior** `p(z|x)`: phân phối của `z` sau khi đã quan sát `x` — thường không tính được trực tiếp.

**Mode coverage là gì?**
Mức độ mô hình sinh bao phủ được các vùng khác nhau (modes) trong phân phối dữ liệu thật. Mode coverage tốt = sinh được đa dạng mẫu, không chỉ tập trung vào một khu vực.

**Sample quality và sample diversity khác nhau thế nào?**
Quality: mỗi mẫu sinh ra trông thật và sắc nét đến đâu. Diversity: mô hình sinh ra nhiều loại mẫu khác nhau hay chỉ lặp lại một vài dạng. Hai yếu tố này thường đánh đổi nhau.

**Tại sao một mô hình có sample đẹp chưa chắc có likelihood tốt?**
GAN có thể tạo ảnh rất sắc nét nhưng chỉ sinh ở một số vùng của không gian (mode collapse), nên likelihood thấp. Ngược lại, autoregressive model có likelihood cao nhưng sample trông mượt mà, đôi khi kém sắc nét theo mắt người.

**Tại sao generative modeling khó hơn classification?**
Classification chỉ cần học ranh giới giữa các lớp. Generative modeling phải học toàn bộ cấu trúc phức tạp của dữ liệu — mọi chi tiết và sự biến đổi trong ảnh người, ngôn ngữ, âm thanh...

---

## 2. Khung xác suất và latent variables

**Latent variable `z` được đưa vào mô hình để làm gì?**
Mô hình hóa các yếu tố ẩn tạo nên dữ liệu: pose, màu sắc, phong cách... Giúp biểu diễn phức tạp `p(x)` dưới dạng đơn giản hơn `p(x) = ∫ p(x|z)p(z) dz`, và học được không gian latent có ý nghĩa.

**Vì sao `p(x) = ∫ p(x|z)p(z) dz` thường khó tính trực tiếp?**
Tích phân trên toàn bộ không gian `z` (thường là liên tục, cao chiều) không có closed-form và quá đắt để tính xấp xỉ bằng Monte Carlo đơn giản.

**Inference trong latent variable model là gì?**
Tính posterior `p(z|x)` — cho biết biến ẩn nào có thể sinh ra điểm dữ liệu `x` đã quan sát. Thường được dùng để encode dữ liệu vào không gian latent.

**Approximate posterior `q(z|x)` là gì?**
Một phân phối xấp xỉ (thường là Gaussian) dùng để ước lượng posterior thật `p(z|x)` vì `p(z|x)` không tính được trực tiếp. Trong VAE, `q(z|x)` là output của encoder network.

**Vì sao posterior thật `p(z|x)` thường không tractable?**
Theo Bayes: `p(z|x) = p(x|z)p(z)/p(x)`. Nhưng `p(x) = ∫ p(x|z)p(z) dz` không tính được (tích phân cao chiều) → mẫu số không biết → posterior không tính được.

**Maximum likelihood training là gì?**
Tối ưu tham số mô hình để maximize xác suất của dữ liệu train: `θ* = argmax_θ Σ log p(x; θ)`. Tương đương với minimize cross-entropy giữa phân phối dữ liệu và phân phối mô hình.

**ELBO là gì? Vì sao phải tối ưu ELBO thay vì log-likelihood trực tiếp?**
ELBO (Evidence Lower BOund) là cận dưới của log-likelihood: `log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))`. Tối ưu ELBO thay vì log p(x) vì log p(x) không tính được trực tiếp, còn ELBO tính được và tối ưu theo gradient descent được.

**Reconstruction term và regularization term trong ELBO có ý nghĩa gì?**
- **Reconstruction** `E_q[log p(x|z)]`: đo decoder tái tạo input tốt đến đâu từ latent `z`.
- **Regularization** `−KL(q(z|x) || p(z))`: ép posterior gần với prior, giúp không gian latent có cấu trúc tốt và có thể sample được.

**KL divergence đo cái gì?**
Đo mức độ khác nhau giữa hai phân phối. `KL(q||p) ≥ 0`, bằng 0 khi và chỉ khi `q = p`. Không đối xứng: `KL(q||p) ≠ KL(p||q)`.

**Vì sao mô hình latent có thể học representation hữu ích?**
Vì latent space phải tổng hợp thông tin cần thiết để tái tạo lại `x`. Các yếu tố quan trọng (pose, màu, shape) phải được encode vào `z`, từ đó `z` trở thành biểu diễn có nghĩa.

**Disentanglement là gì?**
Khi mỗi chiều của latent `z` kiểm soát một yếu tố biến đổi độc lập của dữ liệu (ví dụ một chiều là màu tóc, một chiều là góc mặt). Disentangled latent space dễ hiểu và kiểm soát hơn.

**Posterior collapse là gì?**
Khi encoder trong VAE "bỏ cuộc", tức là `q(z|x) ≈ p(z)` (posterior gần bằng prior) — decoder không dùng `z` nữa và học tái tạo `x` mà không cần thông tin từ latent. Thường xảy ra khi decoder quá mạnh.

---

## 3. Autoregressive models

**Autoregressive model là gì?**
Mô hình sinh dữ liệu từng phần tử một, mỗi phần tử điều kiện trên tất cả phần tử trước đó: `p(x) = ∏ p(x_t | x_1, ..., x_{t-1})`.

**Vì sao có thể viết joint distribution thành product of conditionals?**
Đây là quy tắc xác suất (chain rule): `p(x_1, x_2, ..., x_T) = p(x_1) × p(x_2|x_1) × ... × p(x_T|x_1,...,x_{T-1})` — đúng với mọi phân phối joint, không cần giả định gì thêm.

**PixelRNN/PixelCNN mô hình hóa ảnh như thế nào?**
Xem ảnh như chuỗi pixel theo thứ tự raster (trái-phải, trên-xuống). Mỗi pixel được sinh có điều kiện trên các pixel đã tạo trước. PixelRNN dùng LSTM, PixelCNN dùng masked convolution để đảm bảo không nhìn tương lai.

**Vì sao autoregressive models cho exact likelihood?**
Likelihood được tính là tích của các xác suất conditional — mỗi cái tính được trực tiếp. Không cần xấp xỉ hay bounds.

**Ưu điểm lớn nhất của autoregressive models là gì?**
Exact likelihood — có thể đánh giá và tối ưu hóa log-likelihood một cách chính xác, không cần surrogate objective.

**Vì sao autoregressive sampling thường chậm?**
Phải sinh từng phần tử một theo thứ tự, mỗi bước cần toàn bộ context trước đó → không song song hóa được quá trình sinh. Train thì song song được (teacher forcing), nhưng inference thì tuần tự.

**Teacher forcing trong autoregressive training có vai trò gì?**
Trong khi train, luôn dùng token thật của bước trước (không phải token mô hình đã sinh) làm input cho bước hiện tại → training song song, ổn định hơn.

**Exposure bias là gì trong mô hình sinh chuỗi?**
Khi train dùng token thật (teacher forcing), nhưng khi inference phải dùng token mô hình tự sinh → phân phối khác nhau → lỗi tích lũy qua các bước sinh dài.

**Autoregressive models mạnh ở text hơn ở image theo nghĩa nào?**
Text có thứ tự tự nhiên (trái phải), token ít hơn và rời rạc. Ảnh có nhiều pixel (hàng nghìn), thứ tự raster ít tự nhiên hơn → sampling chậm và khó hơn so với text.

**Vì sao likelihood tốt không đồng nghĩa sample nhìn đẹp hơn GAN?**
Likelihood đánh giá mô hình trên toàn bộ phân phối — kể cả vùng dữ liệu "trung bình". GAN tập trung vào realism của từng sample riêng lẻ. Hai mục tiêu khác nhau dẫn tới đánh đổi khác nhau.

**Khi nào nên chọn autoregressive hơn GAN/VAE?**
Khi cần exact likelihood (density estimation, compression), khi dữ liệu là chuỗi rời rạc tự nhiên (text, audio discrete tokens), hoặc khi cần training ổn định không gặp mode collapse.

**Decoder của VQ-VAE thường kết hợp autoregressive prior theo cách nào?**
VQ-VAE encode ảnh thành chuỗi discrete codes. Sau đó dùng PixelCNN (autoregressive) để học prior trên chuỗi codes đó → sinh ảnh mới bằng cách sample prior rồi decode.

---

## 4. VAE

**VAE là gì?**
Variational Autoencoder: mô hình sinh latent variable học bằng cách tối ưu ELBO. Gồm encoder `q(z|x)` mapping input vào phân phối latent, và decoder `p(x|z)` tái tạo input từ latent sample.

**VAE khác autoencoder thường ở điểm nào?**
Autoencoder thường: encode thành vector điểm tất định, decode lại. Không có ràng buộc xác suất → không sample được latent ngoài training data. VAE: encode thành phân phối `q(z|x)` với mean và variance, có KL regularization → latent space trơn và sample được.

**Encoder và decoder trong VAE học cái gì?**
Encoder: học ánh xạ `x → (μ, σ)` — tham số của phân phối `q(z|x) = N(μ, σ²)`. Decoder: học tái tạo `x` từ sample `z ~ q(z|x)`.

**Prior `p(z)` trong VAE thường chọn ra sao?**
Gaussian chuẩn `N(0, I)` — đơn giản, có thể sample dễ dàng, và KL divergence với Gaussian khác có closed-form.

**ELBO của VAE gồm những thành phần nào?**
`ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))`
- Thành phần 1: reconstruction loss (tái tạo tốt đến đâu).
- Thành phần 2: KL term (posterior gần prior đến đâu).

**Reconstruction loss và KL term đánh đổi điều gì?**
Reconstruction muốn encoder encode chi tiết nhất → `q(z|x)` sắc nét, hẹp. KL muốn `q(z|x)` gần prior `N(0,I)` → rộng hơn. Cân bằng hai term quyết định chất lượng latent space.

**Reparameterization trick là gì?**
Thay vì sample `z ~ N(μ, σ²)` trực tiếp, viết lại: `z = μ + σ × ε`, với `ε ~ N(0, I)`. Gradient chạy qua `μ` và `σ` (deterministic), không qua bước sampling ngẫu nhiên.

**Vì sao sampling `z` trực tiếp lại khó backpropagate?**
Bước sampling `z ~ q(z|x)` là ngẫu nhiên, không có gradient đối với tham số encoder. Không thể backprop qua nút ngẫu nhiên bình thường.

**Reparameterization giúp gradient đi qua stochastic node như thế nào?**
Chuyển phần ngẫu nhiên `ε` ra khỏi đồ thị tính toán — `ε` là externally sampled, không phụ thuộc tham số. Gradient chạy qua `μ` và `σ` một cách bình thường.

**VAE có ưu điểm gì so với GAN?**
Training ổn định hơn (không có game minimax), có exact ELBO làm objective, có encoder để encode dữ liệu vào latent, latent space có cấu trúc tốt, dễ interpolate.

**Vì sao VAE thường cho sample "mượt nhưng hơi blur" trong ảnh?**
Reconstruction loss (thường là MSE) tối ưu theo nghĩa trung bình — tạo ra ảnh "trung bình" giữa nhiều khả năng thay vì một khả năng sắc nét cụ thể. Diffusion và GAN giải quyết vấn đề này tốt hơn.

**Khi nào latent space của VAE đặc biệt hữu ích?**
Khi cần encoder để mapping dữ liệu → vector biểu diễn, để interpolate giữa hai điểm dữ liệu, để disentangle các yếu tố biến đổi, hoặc làm downstream task với embeddings.

**Beta-VAE là gì?**
Tăng hệ số KL term lên `β > 1`: `ELBO = E_q[log p(x|z)] - β × KL(...)`. Tăng áp lực regularization → latent space disentangled hơn nhưng reconstruction quality có thể giảm.

**Posterior collapse trong VAE xảy ra khi nào?**
Khi decoder đủ mạnh để tái tạo `x` mà không cần thông tin từ `z`. Encoder "tắt" (posterior → prior), KL term = 0, decoder trở thành mô hình unconditional. Thường xảy ra với decoder là autoregressive mạnh.

**Vì sao decoder quá mạnh có thể làm latent bị bỏ qua?**
Decoder mạnh có thể model `p(x)` trực tiếp mà không cần `z`. Training sẽ tìm ra rằng bỏ qua `z` cũng đạt được reconstruction tốt → KL term minimize về 0 → posterior collapse.

**VQ-VAE khác VAE chuẩn ở chỗ nào?**
VQ-VAE dùng latent **rời rạc** (discrete) thay vì liên tục. Không có KL term hay reparameterization. Thay vào đó dùng vector quantization: ánh xạ encoder output vào phần tử gần nhất trong một codebook cố định.

---

## 5. VQ-VAE

**VQ-VAE là gì?**
Variational AutoEncoder với latent space rời rạc. Encoder tạo ra vector liên tục, sau đó được "snap" vào phần tử gần nhất trong một codebook → latent là index rời rạc trong codebook.

**VQ-VAE khác VAE ở latent continuous/discrete như thế nào?**
VAE: latent là vector liên tục với phân phối Gaussian. VQ-VAE: latent là index rời rạc trong codebook → có thể dùng autoregressive model trên chuỗi indices đó.

**Vector quantization có vai trò gì?**
Chuyển vector liên tục từ encoder thành phần tử rời rạc trong codebook. Gradient qua bước này dùng straight-through estimator (copy gradient từ decoder về encoder bỏ qua bước quantize).

**Vì sao discrete latent có thể hữu ích cho speech/image/token-like data?**
Dữ liệu như speech hay ảnh có cấu trúc rời rạc tiềm ẩn (phoneme, visual token). Latent rời rạc phù hợp hơn, đặc biệt khi muốn dùng language model trên tokens đó.

**Codebook trong VQ-VAE là gì?**
Tập hợp các embedding vector `{e_1, ..., e_K}` học được. Mỗi encoder output được ánh xạ vào embedding gần nhất trong codebook bằng nearest-neighbor lookup.

**Vì sao VQ-VAE thường đi kèm autoregressive prior?**
Latent của VQ-VAE là chuỗi discrete tokens → có thể học autoregressive prior (PixelCNN, Transformer) trên chuỗi này để sinh ảnh mới bằng cách sample từ prior.

**VQ-VAE giúp giảm posterior collapse theo trực giác nào?**
Không có KL term áp lực như VAE, và latent là rời rạc nên khó bị bỏ qua hoàn toàn. Encoder buộc phải chọn một entry trong codebook → thông tin luôn được dùng.

**Khi nào nên nghĩ tới VQ-VAE thay vì VAE chuẩn?**
Khi muốn latent rời rạc (để dùng với language model, tokenization), khi dữ liệu là speech/audio, hoặc khi muốn tránh posterior collapse của VAE thông thường.

---

## 6. GAN

**GAN là gì?**
Mạng đối nghịch sinh (Generative Adversarial Network): một generator sinh mẫu giả, một discriminator phân biệt thật/giả. Hai mạng "cạnh tranh" nhau trong một game minimax cho đến khi generator sinh được mẫu đánh lừa được discriminator.

**Generator học gì, discriminator học gì?**
Generator: học ánh xạ từ noise `z ~ N(0,I)` sang dữ liệu thật phân phối `p_data`. Discriminator: học phân biệt dữ liệu thật với dữ liệu do generator sinh.

**Objective minimax của GAN có ý nghĩa trực giác gì?**
Discriminator muốn maximize khả năng phân biệt thật/giả. Generator muốn minimize khả năng đó (đánh lừa discriminator). Khi cân bằng Nash đạt được, generator sinh mẫu giống data thật.

**Vì sao GAN không cần explicit likelihood?**
GAN là implicit model — không cần định nghĩa `p(x)` mà chỉ học cách sample từ nó thông qua feedback từ discriminator. Không cần tính tích phân phức tạp.

**GAN có ưu điểm gì về sample quality?**
GAN tạo sample sắc nét, chi tiết cao vì discriminator đánh giá trực tiếp chất lượng từng sample. Không bị "blur" như VAE.

**Vì sao GAN nổi tiếng khó train?**
Phải cân bằng giữa generator và discriminator — nếu một bên mạnh hơn quá nhiều, bên kia không học được. Objective không ổn định, dễ dao động hoặc không hội tụ.

**Mode collapse là gì?**
Generator học cách sinh ra chỉ một vài loại mẫu (vài "mode") trông thật, thay vì toàn bộ phân phối đa dạng — discriminator không phát hiện được nhưng diversity rất thấp.

**Vì sao discriminator quá mạnh hoặc quá yếu đều có thể gây vấn đề?**
Quá mạnh: gradient về generator ≈ 0 (saturate) vì discriminator phân biệt hoàn hảo ngay → generator không học được. Quá yếu: feedback không đáng tin → generator không cải thiện.

**Non-saturating GAN loss là gì và vì sao hay dùng hơn dạng minimax thuần?**
Thay vì `min_G log(1 - D(G(z)))` (bão hòa sớm), dùng `max_G log(D(G(z)))` → gradient mạnh hơn khi generator kém, tránh bị kẹt ở đầu training.

**Conditional GAN là gì?**
GAN có thêm điều kiện `y` (nhãn, text, ảnh...) cho cả generator và discriminator. Generator sinh mẫu theo điều kiện, discriminator đánh giá cặp `(x, y)` có hợp lệ không.

**cGAN dùng điều kiện `y` ở đâu?**
`y` được concat vào input của cả generator (cùng noise `z`) và discriminator (cùng `x`), để cả hai đều nhận điều kiện.

**DCGAN là gì về mặt trực giác kiến trúc?**
Deep Convolutional GAN: dùng conv layers cho discriminator và deconv (transposed conv) layers cho generator. Loại bỏ FC layers, dùng batch norm, sinh ảnh có chất lượng tốt hơn GAN FC.

**Vì sao GAN hay được dùng cho image synthesis và image-to-image translation?**
Sample quality cao, đặc biệt mạnh khi cần ảnh sắc nét và chi tiết. Conditional GAN dễ define objective kiểu "input ảnh A sinh ảnh B". Pix2Pix, CycleGAN là ví dụ điển hình.

**WGAN thay đổi gì so với GAN chuẩn?**
Dùng Wasserstein distance thay cho JS divergence là objective. Critic (không phải discriminator) không bị clip vào [0,1], tính điểm thực sự cho thật/giả, cho gradient ổn định hơn.

**Vì sao Wasserstein distance giúp training "có nghĩa" hơn?**
JS divergence có thể bằng hằng số (log 2) ngay cả khi hai phân phối không chồng lên nhau → gradient = 0. Wasserstein distance vẫn cho gradient có nghĩa dù phân phối không chồng, giúp generator học được.

**Tại sao WGAN dùng "critic" thay vì "discriminator"?**
Vì critic không phải classifier (không phân biệt 0/1) mà tính điểm thực (real-valued score) cho mức độ "thật" của mẫu — không clamp output vào [0,1].

**Gradient penalty trong WGAN-GP để làm gì?**
Ép gradient của critic có norm gần 1 tại mọi điểm (Lipschitz constraint). Thay cho weight clipping trong WGAN gốc — ổn định hơn và không gây vanishing gradient.

**Weight clipping trong WGAN gốc có nhược điểm gì?**
Clip weights của critic vào [-c, c] áp đặt Lipschitz constraint nhưng gây gradient vanish hoặc explode. Mạng muốn dùng hết range [-c, c] → weights tập trung ở hai cực.

**GAN có mạnh cho likelihood estimation không?**
Không. GAN là implicit model — không tính được `p(x)` trực tiếp. Không thể dùng cho density estimation hay compression.

**Khi nào nên chọn GAN thay vì VAE/diffusion?**
Khi cần sample quality cực cao với tốc độ sinh nhanh (single forward pass), khi task là image synthesis/translation và không cần likelihood. Nếu training stability là vấn đề, diffusion có thể tốt hơn.

---

## 7. Normalizing flows

**Normalizing flow là gì?**
Mô hình sinh dùng chuỗi các phép biến đổi khả nghịch (invertible transformations) để ánh xạ từ phân phối đơn giản (Gaussian) sang phân phối phức tạp của dữ liệu, và ngược lại.

**Vì sao flow phải dùng biến đổi khả nghịch?**
Để có thể tính được: (1) log-likelihood qua change-of-variables formula, (2) inference `z` từ `x` (encode), và (3) sampling `x` từ `z` (decode). Cả ba hướng đều cần biến đổi hai chiều.

**Change-of-variables formula nói gì?**
Nếu `x = f(z)` với `f` khả nghịch: `log p(x) = log p(z) + log |det(∂z/∂x)|` — tức là log-density của `x` bằng log-density của `z` cộng với log của Jacobian determinant bù lại cho thay đổi thể tích.

**Jacobian determinant xuất hiện để làm gì?**
Bù cho sự "co giãn" của thể tích không gian khi biến đổi. Nếu `f` "kéo giãn" không gian tại một vùng, mật độ ở đó giảm tương ứng → Jacobian bù lại.

**Vì sao flow cho exact likelihood?**
Vì có thể tính trực tiếp `log p(x)` thông qua change-of-variables. Không cần xấp xỉ hay bounds — Jacobian và prior đều tính được chính xác.

**Vì sao flow cũng cho exact latent inference?**
Vì `f` có nghịch đảo `f^{-1}` → có thể tính chính xác `z = f^{-1}(x)` từ bất kỳ `x` nào, không cần encoder xấp xỉ như VAE.

**Coupling layer là gì?**
Một loại biến đổi khả nghịch tractable: chia `x` thành hai phần `x_1, x_2`. Phần `x_1` đi thẳng, phần `x_2` được biến đổi bằng hàm phụ thuộc `x_1`. Jacobian của coupling layer là tam giác → det tính được hiệu quả.

**Real NVP giải bài toán tractable Jacobian như thế nào?**
Dùng coupling layers: Jacobian có dạng tam giác block → determinant = tích các phần tử đường chéo → tính trong `O(d)` thay vì `O(d³)`. Biến đổi khả nghịch dễ tính.

**Flow mạnh ở điểm nào so với VAE?**
Exact likelihood, exact inference (không cần sparse approximation), và  biểu diễn latent space chính xác. Không có gap giữa likelihood thật và objective tối ưu.

**Flow yếu ở điểm nào so với diffusion/GAN?**
Ràng buộc invertibility giới hạn kiến trúc (không dùng được mọi loại network). Thường cần nhiều tham số hơn để đạt sample quality tương đương GAN hay diffusion.

**Vì sao thiết kế kiến trúc flow bị ràng buộc bởi invertibility?**
Mỗi lớp trong flow phải có nghịch đảo tính được và Jacobian tractable. Không phải mọi cấu trúc mạng đều thỏa mãn điều này → phải dùng kiến trúc đặc biệt như coupling layers, autoregressive flows.

**Khi nào normalizing flows đặc biệt hấp dẫn?**
Khi cần exact likelihood (density estimation, anomaly detection, compression), cần exact inference, hoặc cần latent space trơn với invertible mapping rõ ràng.

---

## 8. Diffusion models

**Diffusion model là gì?**
Mô hình sinh học cách đảo ngược quá trình thêm nhiễu dần dần. Quá trình thuận: thêm Gaussian noise từng bước nhỏ cho đến khi dữ liệu trở thành noise thuần. Quá trình nghịch: học cách khử noise từng bước để tái tạo dữ liệu từ noise.

**Forward diffusion process làm gì?**
Thêm Gaussian noise vào dữ liệu theo lịch trình `T` bước: `x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε`. Sau đủ bước, `x_T ≈ N(0, I)` — dữ liệu bị phá hoàn toàn thành noise.

**Reverse diffusion process làm gì?**
Học cách khử noise dần dần từ `x_T ~ N(0,I)` về dữ liệu `x_0`. Mỗi bước `t`, mô hình dự đoán noise hoặc mean của `p(x_{t-1}|x_t)` để lấy một bước về phía dữ liệu sạch.

**Vì sao mô hình học denoising lại có thể sinh mẫu mới?**
Nếu học được cách đảo ngược từng bước nhỏ của quá trình thêm nhiễu, có thể bắt đầu từ noise thuần `N(0,I)` rồi lặp lại denoising `T` bước → thu được sample giống dữ liệu thật.

**Noise schedule là gì?**
Lịch trình kiểm soát lượng noise thêm vào ở mỗi bước `t` trong forward process (các hệ số `β_t`). Ảnh hưởng đến tốc độ phá hủy dữ liệu và chất lượng học của reverse process.

**Mô hình trong DDPM thường dự đoán noise, sample gốc, hay score?**
Thường dự đoán noise `ε` đã thêm vào tại bước `t` — tức là mô hình `ε_θ(x_t, t)`. Các biến thể khác dự đoán `x_0` hoặc score `∇ log p(x_t)` — về mặt toán học tương đương nhau.

**Denoising score matching liên quan gì đến diffusion?**
Score function `∇_x log p(x)` là gradient của log-density. Diffusion model học score function tại mỗi noise level → kết nối với score-based generative modeling. DDPM và score-based models về cơ bản cùng một framework khi nhìn từ SDE.

**Vì sao diffusion thường cho sample chất lượng rất cao?**
Quá trình tạo mẫu có nhiều bước nhỏ tỉ mỉ, mỗi bước tinh chỉnh dần → kết quả chi tiết, đa dạng. Không bị mode collapse như GAN, không bị blur như VAE.

**Vì sao diffusion sampling chậm hơn GAN?**
Phải lặp `T` bước denoising (thường 50–1000 bước) để sinh một mẫu, mỗi bước là một forward pass qua mạng. GAN chỉ cần một forward pass duy nhất.

**Classifier guidance là gì?**
Dùng gradient của một classifier bên ngoài `log p(y|x_t)` để hướng reverse process về phía class mong muốn. Cần train classifier riêng trên noisy data.

**Classifier-free guidance là gì về trực giác?**
Train mô hình vừa có điều kiện (với label `y`) vừa không điều kiện (không có `y`, dùng null). Khi sinh, kết hợp: `ε_guided = ε_uncond + w × (ε_cond - ε_uncond)`. `w` lớn → bám điều kiện chặt hơn nhưng diversity giảm.

**Conditional diffusion hoạt động thế nào?**
Thêm điều kiện `c` (text, class, ảnh...) vào input của denoising network (thường qua cross-attention với text embedding). Mô hình học `ε_θ(x_t, t, c)` → sinh mẫu theo điều kiện.

**Latent diffusion khác pixel-space diffusion ở đâu?**
Latent diffusion (như Stable Diffusion) chạy diffusion trong latent space của VAE thay vì pixel space. Latent nhỏ hơn nhiều → rẻ hơn nhiều về compute, nhớ được nhiều step hơn trong cùng bộ nhớ.

**DDIM tăng tốc sampling theo ý tưởng nào?**
Rút ra deterministic sampling trajectory cho phép bỏ qua nhiều bước (dùng 50 bước thay vì 1000) bằng cách xấp xỉ ODE thay vì SDE. Chất lượng vẫn tốt với ít bước hơn nhiều.

**Flow matching khác diffusion truyền thống ở mức rất trực giác nào?**
Diffusion: quỹ đạo từ noise → data được định nghĩa qua SDE/noise schedule. Flow matching: học trực tiếp vector field đơn giản (đường thẳng từ noise → data), không cần noise schedule phức tạp, train đơn giản hơn và thường ổn định hơn.

**Khi nào diffusion đáng chọn hơn GAN/VAE?**
Khi cần chất lượng cao + diversity cao + training ổn định. Chấp nhận được việc sinh chậm hơn. Là lựa chọn mặc định cho text-to-image hiện đại.

---

## 9. Energy-based và score-based models

**Energy-based model là gì?**
Mô hình định nghĩa phân phối qua hàm năng lượng `E_θ(x)`: `p_θ(x) = exp(-E_θ(x)) / Z`, trong đó `Z` là hằng số chuẩn hóa (partition function) không biết.

**"Năng lượng thấp" nghĩa là gì về mặt xác suất hoặc trực giác?**
`p(x) ∝ exp(-E(x))` → năng lượng thấp = xác suất cao. Miền dữ liệu thật được mô hình gán năng lượng thấp, còn vùng "không phải data" được gán năng lượng cao.

**Inference trong EBM là làm gì?**
Lấy mẫu từ `p(x)` bằng MCMC (ví dụ Langevin dynamics) vì không có cách direct sampling như GAN hay flow. Phải dùng iterative sampling.

**Vì sao partition function làm training EBM khó?**
`Z = ∫ exp(-E(x)) dx` không tính được trực tiếp nên không tính được gradient của log-likelihood một cách dễ dàng. Phải dùng contrastive divergence, noise contrastive estimation, hoặc các xấp xỉ khác.

**Score function `∇_x log p(x)` là gì?**
Gradient của log-density theo `x` — chỉ hướng tăng xác suất của `p(x)` trong không gian data. Không cần biết `Z` để tính score.

**Score-based model học cái gì thay vì trực tiếp học density?**
Học score function `s_θ(x) ≈ ∇_x log p(x)`. Vì score không cần `Z`, tránh được bài toán partition function. Sau khi học score, lấy mẫu bằng Langevin dynamics.

**Score matching khác maximum likelihood ở đâu?**
Score matching tối ưu để `s_θ(x) ≈ ∇_x log p(x)` — không cần `Z`. Maximum likelihood cần tính `log p(x)` đầy đủ, bao gồm `Z`.

**Score-based models liên hệ với diffusion như thế nào?**
Diffusion model tương đương với học score function ở nhiều noise level: `s_θ(x_t, t) ≈ ∇_{x_t} log p(x_t)`. Reverse process của diffusion dùng score để tính gradient điều hướng về data.

---

## 10. Conditional generation và controllable generation

**Conditional generative model là gì?**
Mô hình học `p(x|c)` thay vì `p(x)` — sinh dữ liệu có điều kiện trên thông tin bổ sung `c` (class label, text prompt, ảnh reference...).

**Làm sao ép mô hình sinh theo class/text/label?**
Thêm điều kiện `c` vào input của mô hình (concat, cross-attention, FiLM conditioning...) và train với cặp `(x, c)`. Mô hình học `p(x|c)`.

**cVAE khác cGAN ở đâu?**
cVAE: encoder/decoder đều nhận điều kiện, có latent space explicit, tối ưu ELBO — ổn định hơn. cGAN: generator/discriminator nhận điều kiện, implicit, sample sắc nét hơn nhưng training không ổn định.

**Conditional autoregressive model dùng điều kiện như thế nào?**
Điều kiện `c` được prepend hoặc cross-attend vào chuỗi, giúp mỗi bước sinh `p(x_t|x_{<t}, c)` phụ thuộc cả điều kiện. Ví dụ: GPT sinh văn bản điều kiện trên system prompt.

**Guidance trong diffusion là một dạng conditional generation ra sao?**
Thay vì điều kiện cứng, guidance điều chỉnh score/noise prediction theo hướng điều kiện trong runtime → linh hoạt, có thể điều chỉnh độ mạnh của điều kiện qua guidance scale.

**Controllability khác diversity như thế nào?**
Controllability: khả năng ép mô hình sinh đúng theo ý muốn. Diversity: sinh ra nhiều kết quả khác nhau khi lặp lại cùng điều kiện. Tăng guidance scale → controllability tăng, diversity giảm.

**Tại sao conditioning tốt nhưng vẫn có thể bị mode dropping?**
Conditioning giúp sinh đúng loại, nhưng trong cùng một điều kiện, mô hình có thể chỉ tạo ra vài dạng phổ biến nhất, bỏ qua các variation hiếm hơn.

**Khi bài toán là text-to-image, vì sao conditioning trở thành trung tâm?**
Vì nhiệm vụ là ánh xạ từ không gian text sang không gian ảnh rộng lớn. Conditioning mechanism quyết định ảnh có "đúng" với text prompt không — đây là thách thức lớn nhất của text-to-image.

---

## 11. Đánh giá mô hình sinh

**Log-likelihood dùng để đánh giá gì?**
Đánh giá mô hình gán xác suất cao cho data thật đến mức nào. Hữu ích cho autoregressive, flows, VAE. Không tính được cho GAN.

**Vì sao với GAN, likelihood thường không thuận tiện?**
GAN là implicit model, không có cách tính `p(x)` trực tiếp → không tính được likelihood.

**FID là gì về trực giác?**
Fréchet Inception Distance: đo khoảng cách giữa phân phối feature của ảnh thật và ảnh sinh (dùng Inception network để extract features). FID thấp = sinh gần với phân phối thật về cả quality lẫn diversity.

**Inception Score đo điều gì và thiếu gì?**
Đo: sample phải rõ ràng (p(y|x) có entropy thấp) và đa dạng (p(y) có entropy cao). Thiếu: không so sánh với dữ liệu thật → mô hình tạo sample đẹp nhưng không như thật vẫn có thể đạt IS cao.

**Precision và recall cho generative models khác nhau thế nào?**
Precision: tỉ lệ sample sinh nằm trong vùng data thật (sample quality). Recall: tỉ lệ data thật được bao phủ bởi phân phối model (diversity). Cả hai cần tốt để đánh giá đầy đủ.

**Sample quality và mode coverage có thể trade-off ra sao?**
Tăng guidance scale → quality cao hơn nhưng diversity giảm (ít mode được bao phủ). Giảm guidance → diversity tăng nhưng có thể quality thấp hơn.

**Tại sao human evaluation vẫn còn được dùng?**
Các metric tự động không phản ánh hoàn toàn nhận thức của con người. Realism, aesthetics, coherence với điều kiện — con người đánh giá tốt hơn metric tự động với nhiều khía cạnh tinh tế.

**Vì sao so sánh hai mô hình sinh luôn phụ thuộc task?**
Một mô hình có thể tốt về FID (diversity + quality tổng thể) nhưng kém về sample chất lượng cao (IS) hay ngược lại. Không có metric nào "tất cả trong một".

**Nén dữ liệu/inpainting/super-resolution có phải là cách đánh giá không?**
Có thể dùng như downstream tasks để đánh giá indirectly. Tuy nhiên hiệu quả trên downstream chưa phản ánh hoàn toàn chất lượng mô hình sinh tổng quát.

**Vì sao "nhìn ảnh đẹp" không đủ để kết luận mô hình tốt?**
Mode collapse có thể tạo ra vài ảnh đẹp nhưng không đa dạng. Mô hình có thể memorize training data. Cần đánh giá trên tập lớn, đo diversity, và kiểm tra generalization.

---

## 12. Thực hành và failure modes

**Vì sao generative models nhạy với optimizer và learning rate?**
Objective của generative models (minimax, ELBO) phức tạp hơn supervised loss, dễ bị local optimum, exploding gradient, hay mất cân bằng. Learning rate ảnh hưởng trực tiếp đến ổn định training.

**Vì sao latent dimension quá nhỏ hoặc quá lớn đều có thể hại?**
Quá nhỏ: không đủ chỗ encode đặc trưng → reconstruction kém. Quá lớn: latent space thưa, khó học prior, dễ overfit, interpolation không mượt.

**Data quality ảnh hưởng mô hình sinh mạnh hơn classification như thế nào?**
Generative model phải học toàn bộ phân phối, bao gồm mọi chi tiết. Ảnh bẩn, caption sai, duplicates → model học phân phối sai. Classification chỉ cần học ranh giới, ít nhạy hơn với noise.

**Memorization trong generative models là gì?**
Mô hình "học thuộc" một số mẫu training cụ thể thay vì học phân phối tổng quát → khi generate ra đúng ảnh training mà không có sáng tạo mới.

**Overfitting trong mô hình sinh biểu hiện ra sao?**
Likelihood tốt trên train nhưng kém trên test set. Sample sinh chỉ gần với training data, không có diversity. FID tốt trên train data distribution nhưng kém trên held-out set.

**Prompt leakage / training data leakage liên hệ với memorization thế nào?**
Mô hình có thể "sinh lại" training data nếu được prompt tương tự. Là vấn đề privacy khi training data chứa thông tin nhạy cảm.

**Tại sao augmentation có thể giúp hoặc làm sai phân phối thật?**
Augmentation thêm variation giúp mô hình tổng quát. Nhưng augmentation sai (thay màu quá mức, crop sai) có thể làm mô hình học phân phối không đúng với thực tế.

**Khi sample lặp lại nhiều, bạn nghi ngờ điều gì?**
Mode collapse (GAN) hoặc memorization — mô hình tập trung vào ít vùng trong không gian. Cần tăng diversity, kiểm tra diversity metric.

**Khi mô hình chỉ sinh ra vài mode, đó là dấu hiệu của gì?**
Mode collapse — đặc biệt phổ biến với GAN. Cần điều chỉnh training (WGAN-GP, minibatch discrimination) hoặc chuyển sang diffusion/flow.

**Khi loss đẹp nhưng mẫu xấu, nên kiểm tra gì trước?**
Với GAN: xem discriminator có over-powerful không, có mode collapse không. Với VAE: reconstructon quality và KL weight. Xem sample diversity, FID, visualize nhiều mẫu.

---

## 13. Phân biệt

**Generative vs discriminative:** Generative học `p(x)` hoặc `p(x,y)`. Discriminative học `p(y|x)`. Generative có thể sinh mẫu mới, discriminative chỉ phân loại.

**Explicit density vs implicit density:** Explicit tính được `p(x)` (flow, VAE, autoregressive). Implicit chỉ sample được (GAN) — không tính được likelihood.

**Likelihood-based vs adversarial training:** Likelihood-based tối ưu log-likelihood hoặc ELBO trực tiếp. Adversarial (GAN) tối ưu qua feedback từ discriminator — không liên quan trực tiếp đến likelihood.

**VAE vs autoencoder thường:** VAE có latent là phân phối, có KL regularization, latent space trơn → interpolate và sample được. AE thường: latent tất định, không sample được.

**VAE vs GAN:** VAE: ổn định, latent explicit, sample hơi blur. GAN: không ổn định, implicit, sample sắc nét, dễ mode collapse.

**GAN vs diffusion:** GAN: nhanh (1 step), chất lượng cao, khó train. Diffusion: chậm (nhiều bước), chất lượng rất cao, diversity tốt, train ổn định.

**Autoregressive vs diffusion:** Autoregressive: sinh tuần tự từng token, exact likelihood. Diffusion: sinh song song nhiều bước khử noise, sample chất lượng cao hơn cho ảnh.

**Flow vs VAE:** Flow: exact likelihood, exact inference. VAE: xấp xỉ qua ELBO, posterior không exact. Flow kiến trúc bị ràng buộc hơn.

**Continuous latent vs discrete latent:** Continuous (VAE): dễ sample, dễ interpolate. Discrete (VQ-VAE): phù hợp với token-based model, dùng được với LM autoregressive prior.

**Sample quality vs sample diversity:** Quality: mỗi sample trông thật. Diversity: sinh được nhiều loại sample khác nhau. Thường có đánh đổi — tăng cái này có thể giảm cái kia.

**Mode collapse vs posterior collapse:** Mode collapse (GAN): generator chỉ sinh vài loại mẫu, bỏ qua nhiều mode. Posterior collapse (VAE): encoder bỏ qua `z`, decoder tự mô hình hóa `p(x)` mà không dùng latent.

**Conditional vs unconditional generation:** Conditional: sinh `p(x|c)` theo điều kiện cho trước. Unconditional: sinh `p(x)` không có thông tin bổ sung — ít kiểm soát hơn.

---

## 14. Câu tự luận

**Trình bày các họ deep generative models phổ biến và điểm khác nhau cốt lõi:**

| Họ | Likelihood | Sample quality | Tốc độ sinh | Ổn định train |
|---|---|---|---|---|
| Autoregressive | Exact | Tốt | Chậm (tuần tự) | Ổn định |
| VAE | ELBO | Trung bình (blur) | Nhanh | Ổn định |
| Flow | Exact | Trung bình | Nhanh | Ổn định |
| GAN | Không có | Rất tốt | Rất nhanh | Khó |
| Diffusion | Xấp xỉ | Tốt nhất | Chậm (nhiều bước) | Ổn định |

**Giải thích ELBO trong VAE và vai trò của reparameterization trick:**
Log-likelihood `log p(x)` không tính được trực tiếp vì phải tích phân qua `z`. ELBO là cận dưới: `ELBO = E_q[log p(x|z)] - KL(q||p)`. Tối ưu ELBO → gián tiếp tăng log-likelihood. Reparameterization: thay sampling `z ~ N(μ,σ)` bằng `z = μ + σε` với `ε ~ N(0,I)` → gradient chạy qua `μ,σ` bình thường, cho phép backprop qua encoder.

**Trình bày cơ chế huấn luyện GAN và vì sao GAN khó train:**
Train luân phiên: (1) fix G, train D phân biệt real/fake. (2) fix D, train G để D phân loại sai. Lý tưởng là Nash equilibrium. Thực tế khó vì: nếu D quá mạnh → gradient về G = 0; nếu G chỉ sinh một mode → không bị phạt → mode collapse. Cân bằng training rate giữa hai mạng là nghệ thuật.

**Giải thích vì sao normalizing flows có exact likelihood:**
Flows dùng chuỗi biến đổi `x = f_K(f_{K-1}(...f_1(z)))`. Mỗi `f_i` là bijection. Theo change-of-variables: `log p(x) = log p(z) + Σ log|det J_i|`. Cả `log p(z)` lẫn từng `log|det J_i|` đều tính được → exact log-likelihood.

**Trình bày trực giác của diffusion models dưới góc nhìn thêm nhiễu rồi khử nhiễu:**
Forward: thêm dần noise vào ảnh qua `T` bước nhỏ → ảnh trở thành white noise. Reverse: train network để khử noise từng bước → học `p(x_{t-1}|x_t)`. Để sinh ảnh mới: bắt đầu từ `z_T ~ N(0,I)`, lặp `T` bước denoising → ảnh mới không có trong training.

**So sánh VAE, GAN, flow, autoregressive, diffusion:**
Xem bảng trên. Diffusion hiện là lựa chọn mạnh nhất về quality + diversity. GAN nhanh nhất. Flow và autoregressive có likelihood. VAE đơn giản và latent space tốt nhất cho downstream tasks.

**Vì sao đánh giá mô hình sinh khó hơn đánh giá mô hình phân loại:**
Không có "ground truth" đúng/sai cho mỗi sample sinh. FID đo phân phối tổng thể nhưng không hoàn hảo. Likelihood tốt ≠ sample đẹp. Diversity và quality đánh đổi nhau. Cần nhiều metric kết hợp và thường phải có human evaluation.

**Nếu cần mô hình sinh cho ảnh/text/speech, chọn họ nào và vì sao:**
- Ảnh chất lượng cao: Diffusion (ổn định, diversity cao, quality tốt nhất hiện tại).
- Ảnh cần sinh nhanh: GAN hoặc distilled diffusion.
- Text: Autoregressive (GPT-style) — phù hợp tự nhiên với token sequence, exact likelihood.
- Speech: Flow-based (WaveGlow, WaveFlow) hoặc diffusion (DiffWave) — exact timing, high quality.
