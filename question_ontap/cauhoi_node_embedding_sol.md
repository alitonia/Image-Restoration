# Giải Đáp Câu Hỏi Node Embedding

## 1. Câu nền tảng

**Node embedding là gì?**
Ánh xạ mỗi node trong graph thành một vector số thực chiều thấp, sao cho thông tin quan trọng của graph (cấu trúc, quan hệ giữa các node) được bảo toàn trong không gian vector đó.

**Vì sao cần ánh xạ node sang vector thấp chiều?**
Để có thể áp dụng các thuật toán ML truyền thống (classifier, regressor…) vốn cần input dạng vector cố định. Không gian vector thấp chiều giúp học hiệu quả và tổng quát hóa tốt hơn.

**Node embedding dùng cho những task nào?**
Node classification, link prediction, community detection, graph clustering, similarity search, visualization.

**Node embedding khác feature engineering thủ công ở điểm nào?**
Feature engineering: người thiết kế tạo các đặc trưng theo domain (degree, clustering coefficient…). Node embedding: học tự động biểu diễn từ cấu trúc graph, không cần can thiệp thủ công.

**Node embedding khác graph embedding như thế nào?**
Node embedding: vector cho từng node. Graph embedding: vector đại diện cho toàn bộ graph (dùng cho graph-level tasks như phân loại phân tử).

**Node embedding khác node classification như thế nào?**
Node embedding: quá trình học biểu diễn (unsupervised hoặc self-supervised) — trung gian trước khi làm task. Node classification: task cuối (có nhãn), thường dùng node embedding như input.

**Một embedding tốt cần giữ lại loại thông tin gì của graph?**
Tùy mục tiêu: cấu trúc láng giềng (local proximity), vai trò cấu trúc (structural equivalence), hoặc cộng đồng (community membership). Không có embedding "tốt nhất" cho mọi task.

**Vì sao các node "giống nhau" nên có embedding gần nhau?**
Vì downstream task thường dựa trên similarity trong embedding space (cosine similarity, dot product). Node gần nhau trong embedding sẽ được xử lý tương tự nhau bởi classifier/regressor.

**"Node similarity" có thể được định nghĩa theo những cách nào?**
- Adjacent (kết nối trực tiếp).
- Đồng xuất hiện trong random walk (cùng context trong walk).
- Structural equivalence (có cùng vai trò cấu trúc dù ở xa nhau).
- Community-based (cùng cộng đồng).

**Khi nào chỉ dùng cấu trúc graph là đủ, khi nào cần thêm node features?**
Chỉ cấu trúc: khi graph không có features (chỉ có cạnh) hoặc topology là thông tin chính. Cần thêm features: khi nodes có thuộc tính quan trọng (phân tử có loại nguyên tử, mạng xã hội có user profile).

---

## 2. Khung encoder-decoder

**Encoder trong node embedding là gì?**
Hàm ánh xạ node `v` thành vector: `ENC(v) = z_v`. Trong shallow embedding (DeepWalk, node2vec): encoder là lookup table — mỗi node có embedding riêng. Trong GNN: encoder là hàm học đặc trưng từ neighborhood.

**Decoder trong node embedding là gì?**
Hàm nhận hai embeddings và tính điểm similarity: `DEC(z_u, z_v) ≈ similarity(u, v)`. Ví dụ: dot product `z_u · z_v` hoặc sigmoid của dot product.

**Vì sao phải định nghĩa một notion of similarity trước khi học embedding?**
Objective của training là làm cho embedding phản ánh đúng similarity đã chọn. Nếu không định nghĩa rõ "giống nhau" nghĩa là gì, không có gì để tối ưu.

**Similarity có thể dựa trên adjacency trực tiếp hay đồng xuất hiện trong random walk?**
Cả hai. Adjacency: hai node kết nối trực tiếp nên có embedding gần nhau. Random walk: hai node hay xuất hiện gần nhau trong walk → share context → embedding gần nhau, kể cả node không trực tiếp kết nối.

**Objective của node embedding thường tối ưu cái gì?**
Tối đa hóa xác suất tái tạo lại similarity thật sự. Ví dụ: tối đa `log σ(z_u·z_v)` cho positive pairs, tối thiểu cho negative pairs.

**Vì sao nhiều phương pháp học embedding theo kiểu unsupervised/self-supervised?**
Nhãn bổ sung cho từng node thường khan hiếm. Cấu trúc graph tự nó là tín hiệu tự nhiên (positive/negative pairs từ random walks) — không cần nhãn người gán.

**Positive pair và negative pair là gì?**
Positive pair: cặp node hay xuất hiện gần nhau (kết nối, đồng xuất hiện trong walk) — objective tối đa similarity. Negative pair: cặp node không liên quan — objective tối thiểu similarity.

**Negative sampling dùng để làm gì?**
Với mỗi positive pair, sample ngẫu nhiên một số node không liên quan làm negative — huấn luyện decoder phân biệt cặp liên quan với cặp ngẫu nhiên. Tránh tính toán trên toàn bộ graph.

**Embedding dimension `d` ảnh hưởng thế nào?**
Quá nhỏ: không đủ capacity để biểu diễn. Quá lớn: overfitting, tốn bộ nhớ và compute. Thường chọn `d = 64–256`.

---

## 3. Random-walk based embeddings

**Vì sao random walk lại hữu ích cho node embedding?**
Walk từ một node sẽ ưu tiên đi qua các node gần (láng giềng và láng giềng của láng giềng). Các node xuất hiện cùng trong walk có xác suất cao là structurally related → dùng đồng xuất hiện trong walk như tín hiệu similarity.

**Tại sao có thể xem random walk như "sentence" trong NLP?**
Chuỗi node trong walk tương tự chuỗi từ trong câu. Áp Word2Vec (SkipGram) lên chuỗi node → học node embedding tương tự word embedding.

**Context window trong graph embedding nghĩa là gì?**
Với mỗi node `v` trong walk, các node xuất hiện trong vòng `w` bước xung quanh `v` là "context" của `v`. Embedding của `v` được học để predict hoặc được predicted từ context nodes.

**Vì sao đồng xuất hiện trong walk phản ánh quan hệ cấu trúc?**
Node gần nhau trong graph sẽ hay xuất hiện cùng trong walk. Tần suất đồng xuất hiện cao → similarity cao → embedding gần nhau.

**Random walk giúp bắt local structure hay global structure?**
Chủ yếu local: walk ưu tiên neighborhood gần. Nhưng với walk dài và nhiều walk → thông tin global cũng lan dần. node2vec có thể điều chỉnh giữa local và global qua `p, q`.

**Walk length và number of walks ảnh hưởng thế nào?**
Walk dài hơn: context rộng hơn, bắt quan hệ xa hơn. Nhiều walk hơn: sampling phong phú hơn, ước lượng similarity chính xác hơn. Tăng cả hai → tốn thêm compute.

---

## 4. DeepWalk

**Ý tưởng chính của DeepWalk là gì?**
Sinh random walks trên graph, xem mỗi walk như một câu, rồi áp SkipGram (Word2Vec) để học embedding: node hay xuất hiện gần nhau trong walk → embedding gần nhau.

**DeepWalk học embedding bằng cách nào?**
Với mỗi node `v`, dùng embedding của `v` để predict các context nodes trong window: tối đa `Σ_{c ∈ context(v)} log P(c | v)` qua SkipGram với negative sampling.

**DeepWalk mượn ý tưởng nào từ NLP?**
Word2Vec / SkipGram: học word embedding từ đồng xuất hiện trong cửa sổ ngữ cảnh. DeepWalk áp y chang với node embedding từ đồng xuất hiện trong random walk.

**Vì sao DeepWalk không cần handcrafted graph features?**
Tự động học từ cấu trúc graph thông qua random walk — không cần người thiết kế thủ công các đặc trưng như clustering coefficient, betweenness centrality…

**Hạn chế lớn của DeepWalk là gì?**
Transductive: mỗi node có embedding riêng — không generalize sang node mới. Không dùng node features. Walk là uniform random → không kiểm soát được loại neighborhood được explore.

**DeepWalk thường mang tính transductive ở điểm nào?**
Mỗi node được gán một embedding vector qua lookup table. Node mới (không có trong training) không có embedding — phải retrain toàn bộ.

**DeepWalk có dùng node attributes không?**
Không. Chỉ dùng cấu trúc topology (cạnh) để sinh random walk. Không tích hợp thông tin attribute của node.

---

## 5. node2vec

**node2vec khác DeepWalk ở điểm nào?**
Thay uniform random walk bằng biased random walk: dùng hai tham số `p` và `q` để kiểm soát hướng đi của walk — có thể điều chỉnh giữa BFS-like và DFS-like exploration.

**"Biased random walk" trong node2vec là gì?**
Khi đang ở node `v` đến từ node `t`, xác suất di chuyển đến node `x` tiếp theo phụ thuộc: khoảng cách của `x` với `t` (gần về lại hay đi xa). Điều chỉnh qua `p` (return) và `q` (in-out).

**Hai tham số `p` và `q` điều khiển điều gì?**
`p` (return parameter): kiểm soát xác suất quay lại node vừa thăm (nhỏ → ít quay lại). `q` (in-out parameter): kiểm soát outward (`q<1` → DFS-like, xa hơn) vs inward (`q>1` → BFS-like, gần hơn).

**Trực giác BFS-like và DFS-like walk trong node2vec là gì?**
BFS-like (`q>1`): khảo sát neighborhood gần → bắt local community structure. DFS-like (`q<1`): đi xa dần → bắt structural role (node có cùng vai trò dù xa nhau).

**Khi nào nên thiên về local neighborhood?**
Khi muốn embedding phản ánh membership trong cùng community/cluster — hai node trong cùng cộng đồng nên có embedding gần nhau. Dùng `q>1`.

**Khi nào nên thiên về structural role similarity?**
Khi muốn hai node có cùng vai trò cấu trúc (ví dụ: cả hai là hub, cả hai là bridge) có embedding gần nhau, dù chúng ở xa nhau trong graph. Dùng `q<1`.

**Vì sao node2vec thường mạnh hơn DeepWalk trên nhiều task?**
Linh hoạt hơn: `p, q` cho phép điều chỉnh loại similarity được bảo toàn. Với dataset và task cụ thể, có thể tune `p, q` để optimize — DeepWalk không có cơ chế này.

**node2vec không nhược điểm gì khi graph quá lớn hoặc thay đổi liên tục?**
Vẫn transductive. Biased random walk tốn compute hơn uniform walk (cần precompute transition probs). Graph thay đổi → phải sinh lại walk và retrain.

---

## 6. LINE

**LINE là gì?**
Large-scale Information Network Embedding: học embedding bảo toàn cả first-order proximity (kết nối trực tiếp) và second-order proximity (share nhiều neighbor chung), thiết kế để scale tới graph rất lớn.

**First-order proximity là gì?**
Hai node trực tiếp kết nối nhau bởi một cạnh → nên có embedding gần nhau. Phản ánh quan hệ trực tiếp.

**Second-order proximity là gì?**
Hai node share nhiều neighbor chung → có vai trò tương tự trong network → nên có embedding gần nhau. Phản ánh "bạn của bạn tôi cũng gần giống tôi".

**Vì sao phải giữ cả first-order lẫn second-order proximity?**
First-order bắt quan hệ trực tiếp nhưng bỏ qua cộng đồng chia sẻ. Second-order bắt structural similarity qua neighborhood chung. Kết hợp cả hai → embedding toàn diện hơn.

**LINE khác DeepWalk/node2vec ở điểm nào?**
LINE không dùng random walk. Tối ưu trực tiếp proximity objectives: maximize similarity cho adjacent (first-order) và cho nodes với shared neighbors (second-order).

**LINE mạnh ở scalability như thế nào?**
Dùng edge sampling và negative sampling → mỗi update chỉ liên quan một edge → scale tốt trên graph hàng triệu node và cạnh. Thiết kế explicit cho large-scale.

**Khi nào LINE phù hợp hơn random-walk methods?**
Khi graph có hướng và/hoặc có trọng số, khi graph rất lớn, và khi muốn tách biệt tường minh hai loại proximity thay vì mix lẫn trong walk.

**Hạn chế của LINE là gì?**
Như DeepWalk: transductive, không dùng node features. Sensitive với hyperparameters. Second-order objective cần thêm tính toán so với first-order.

---

## 7. Transductive vs inductive và liên hệ với GNN

**Transductive node embedding là gì?**
Học embedding cho từng node cụ thể trong training graph. Kết quả là bảng lookup `{node_id → vector}`. Không generalize: node mới → cần retrain.

**Inductive node embedding là gì?**
Học một hàm `f(node features, neighborhood) → embedding`. Áp hàm này cho node mới mà không cần retrain — chỉ cần biết features và neighbors của node mới.

**Vì sao DeepWalk/node2vec thường là transductive?**
Embedding của mỗi node là tham số riêng trong lookup table — không có hàm chung để map node mới.

**Vì sao GraphSAGE được xem là inductive?**
GraphSAGE học aggregation function (dùng MLP/LSTM/pooling) → apply function lên neighborhood của bất kỳ node nào, kể cả node chưa thấy lúc train.

**"Learn embeddings" khác gì với "learn an embedding function"?**
Learn embeddings: học vector cụ thể cho từng node → transductive. Learn embedding function: học hàm `f` map features + structure → embedding → inductive, tổng quát được.

**Khi graph có node mới liên tục, vì sao shallow embedding bất tiện?**
Mỗi lần có node mới phải chạy lại optimization → tốn thời gian. Shallow embedding không tổng quát hóa tự động.

**Node embedding cổ điển khác GNN-based embedding ở điểm nào?**
Cổ điển (DeepWalk, node2vec, LINE): shallow lookup table, chỉ dùng structure, transductive. GNN: dùng cả structure lẫn features, học hàm tổng hợp → inductive, expressive hơn.

**Khi nào nên dùng node2vec thay vì GraphSAGE?**
Khi không có node features, khi muốn thuật toán đơn giản và nhanh, khi graph tĩnh (không thay đổi), khi task chỉ cần similarity dựa trên structure.

**Khi nào node features làm thay đổi hoàn toàn lựa chọn phương pháp?**
Khi features mang thông tin quan trọng hơn structure (ví dụ: phân tử — loại nguyên tử quan trọng hơn topology), GNN là lựa chọn bắt buộc. Shallow methods bỏ hoàn toàn thông tin này.

---

## 8. Huấn luyện và đánh giá

**Node embedding thường được đánh giá bằng task downstream nào?**
Node classification (gán nhãn node), link prediction (dự đoán cạnh), clustering (community detection), visualization (t-SNE/UMAP của embeddings).

**Vì sao node classification hay được dùng để đánh giá chất lượng embedding?**
Đơn giản: train classifier (logistic regression/SVM) trên embeddings → accuracy phản ánh chất lượng thông tin trong embedding. Không cần task-specific architecture.

**Link prediction đánh giá embedding như thế nào?**
Score cạnh có thật cao hơn cạnh không tồn tại: `score(u,v) = z_u · z_v`. Dùng AUC hoặc Average Precision trên tập test edges.

**Visualization bằng t-SNE/UMAP có đủ để kết luận embedding tốt không?**
Không. Visualization chỉ 2D — mất nhiều thông tin. Dùng để kiểm tra sơ bộ (node cùng class có cluster lại không), nhưng phải kết hợp với quantitative metrics.

**Vì sao split train/test trên graph dễ bị leakage?**
Graph là một cấu trúc liên kết — test nodes có thể kết nối với train nodes → thông tin về test nodes rò rỉ qua edges vào training data qua message passing.

**Embedding dimension chọn quá lớn có thể gây gì?**
Overfitting: embedding fit perfect vào training pairs nhưng kém generalization. Tốn bộ nhớ, chậm hơn. Thường 64–256 là đủ cho hầu hết graph.

**Negative sampling distribution ảnh hưởng gì tới kết quả?**
Sample uniform: nhiều node degree thấp → model không học phân biệt tốt với node phổ biến. Sample theo degree (như Word2Vec): node phổ biến xuất hiện nhiều hơn làm negative → model học tốt hơn với hub nodes.

**Một embedding tốt cho classification có chắc tốt cho link prediction không?**
Không nhất thiết. Classification cần embedding phân biệt theo nhãn. Link prediction cần embedding nắm bắt quan hệ kết nối. Hai objective khác nhau đôi khi dẫn tới embedding khác nhau.

**Vì sao graph động làm việc đánh giá khó hơn?**
Cạnh thêm/xóa theo thời gian → embedding cần cập nhật. Evaluation trên graph tại thời điểm T có thể khác với T+1. Cần đánh giá trên temporal splits.

**Khi embedding học ra chỉ phản ánh degree, đó là dấu hiệu của vấn đề gì?**
Model collapse: không học được thông tin phong phú hơn degree. Có thể do: quá ít walk, window quá nhỏ, negative sampling kém, hoặc objective quá đơn giản.

---

## 9. Phân biệt

**Node embedding vs graph embedding:** Node: vector cho từng node riêng. Graph: vector cho cả graph — dùng weighted sum/pooling của node embeddings.

**Feature engineering vs representation learning:** Feature engineering: tạo thủ công, domain knowledge. Representation learning: tự học từ data — linh hoạt hơn, ít cần expert.

**Encoder vs decoder:** Encoder: node → vector. Decoder: vector pair → similarity score. Train encoder để decoder cho similarity score đúng.

**DeepWalk vs node2vec:** DeepWalk: uniform random walk. node2vec: biased random walk với `p, q` → linh hoạt hơn, tune được loại similarity.

**node2vec vs LINE:** node2vec: random walk + SkipGram. LINE: tối ưu trực tiếp first/second order proximity — không cần walk, scale tốt hơn.

**First-order vs second-order proximity:** First-order: kết nối trực tiếp. Second-order: share nhiều neighbor chung — gián tiếp hơn.

**Transductive vs inductive:** Transductive: embedding cố định cho mỗi node, không generalize. Inductive: học hàm embedding → áp cho node mới.

**Shallow embedding vs GNN embedding:** Shallow: lookup table, chỉ structure. GNN: học hàm từ features + structure — inductive, expressive hơn.

**Structural similarity vs neighborhood proximity:** Structural: node cùng vai trò cấu trúc dù xa. Neighborhood: node gần nhau trong walk/kết nối.

**Unsupervised embedding vs supervised node classification:** Unsupervised: học embedding không cần nhãn. Supervised: dùng nhãn trực tiếp để train classifier trên graph.

---

## 10. Câu tự luận

**Trình bày khung encoder-decoder trong node embedding:**
Encoder ánh xạ node `v` → vector `z_v`. Decoder nhận cặp `(z_u, z_v)` → similarity score (thường là dot product). Objective: tối đa decoder score cho positive pairs (nodes nên similar), tối thiểu cho negative pairs. Tối ưu bằng SGD với negative sampling.

**Giải thích vì sao random walk có thể dùng để học node embedding:**
Random walk tạo ra chuỗi node với xác suất tỉ lệ với cấu trúc graph. Nodes gần nhau structurally hay đồng xuất hiện trong walk → xem như "context words" → áp Word2Vec SkipGram: embedding được cập nhật để predict context → nodes co-occurring có embedding gần nhau.

**So sánh DeepWalk, node2vec, và LINE:**
DeepWalk: đơn giản nhất, uniform walk, không tune được.
node2vec: biased walk với `p,q`, linh hoạt trade-off local↔structural.
LINE: không dùng walk, optimize proximity trực tiếp, tốt nhất cho graph lớn có hướng/trọng số.

**Vì sao node2vec linh hoạt hơn DeepWalk?**
Tham số `p, q` cho phép điều chỉnh: `q<1` → DFS (structural roles), `q>1` → BFS (community). Có thể tune hyperparameter cho từng task/graph cụ thể để tối ưu downstream performance.

**Vì sao các shallow node embedding methods thường mang tính transductive?**
Mỗi node có embedding riêng là tham số của model. Để sinh embedding cho node mới phải thêm tham số mới và tối ưu lại. Không có "công thức chung" để map node mới → embedding.

**Khi nào nên chuyển từ node embedding cổ điển sang GNN/GraphSAGE?**
Khi: (1) nodes có features quan trọng muốn tích hợp; (2) graph thay đổi liên tục và cần inductive; (3) cần expressive power cao hơn (phân biệt cấu trúc phức tạp); (4) task cần end-to-end optimization với downstream loss.
