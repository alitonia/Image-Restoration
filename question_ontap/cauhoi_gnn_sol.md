# Giải Đáp Câu Hỏi Graph Neural Networks

## 1. Câu nền tảng

**Graph là gì? Node, edge, adjacency matrix là gì?**
Graph `G = (V, E)` gồm tập nút (node) và tập cạnh (edge). Adjacency matrix `A`: `A[i][j] = 1` nếu tồn tại cạnh giữa node `i` và `j`, ngược lại = 0.

**Node features, edge features, graph features khác nhau thế nào?**
Node features: vector đặc trưng của từng nút (ví dụ: tuổi, loại nguyên tử). Edge features: đặc trưng của mỗi cạnh (ví dụ: loại liên kết hoá học, trọng số). Graph features: đặc trưng toàn bộ graph (ví dụ: số node, density).

**Vì sao dữ liệu graph khó hơn image/text?**
Không có kích thước cố định, không có thứ tự tự nhiên giữa các node, kết nối không đều (degree khác nhau), không regular như pixel grid hay token sequence.

**Khi nào bài toán là node classification?**
Khi muốn gán nhãn cho từng node: phân loại người dùng mạng xã hội, phân loại loại protein trong mạng tương tác.

**Khi nào là link prediction?**
Khi muốn dự đoán có tồn tại cạnh giữa hai node không: gợi ý bạn bè, dự đoán tương tác protein-protein, knowledge graph completion.

**Khi nào là graph classification / graph regression?**
Khi toàn bộ graph là một mẫu dữ liệu: phân loại phân tử (độc hay không), dự đoán tính chất vật liệu, phân loại circuit.

**GNN khác graph embedding kiểu DeepWalk/Node2Vec ở điểm nào?**
DeepWalk/Node2Vec: học embedding tĩnh cho mỗi node — không tổng quát sang node mới. GNN: học hàm tổng hợp đặc trưng từ cấu trúc và features → có thể áp dụng cho node mới (inductive).

**GNN tận dụng cả cấu trúc graph lẫn features như thế nào?**
GNN truyền thông điệp theo cạnh của graph (cấu trúc) và tổng hợp features từ các node láng giềng → embedding của mỗi node phụ thuộc cả vị trí trong graph lẫn features của nó và láng giềng.

---

## 2. Message passing

**"Message passing" trong GNN là gì?**
Cơ chế mỗi node tổng hợp thông tin từ các node láng giềng của nó: (1) mỗi node tính "thông điệp" gửi cho láng giềng, (2) mỗi node nhận và tổng hợp thông điệp từ láng giềng, (3) cập nhật embedding của chính nó.

**Một layer GNN thường gồm những bước nào?**
Message: `m_{u→v} = M(h_u, h_v, e_{uv})`. Aggregate: `a_v = AGG({m_{u→v} | u ∈ N(v)})`. Update: `h_v' = U(h_v, a_v)`.

**Vì sao phép aggregate phải bất biến theo thứ tự lân cận?**
Graph không có thứ tự tự nhiên giữa các láng giềng — node `u` có thể có láng giềng theo thứ tự bất kỳ. Phép aggregate phải cho kết quả như nhau dù thứ tự nào (sum, mean, max đều permutation-invariant).

**Mean / sum / max aggregation khác nhau thế nào?**
Mean: trung bình — bình thường hóa theo degree, không phân biệt được tập khác nhau nhưng cùng mean. Sum: tổng — nhạy với kích thước neighborhood, phân biệt tốt hơn. Max: chỉ giữ feature nổi bật nhất — mất thông tin về số lượng.

**Sau `k` layer thì một node "nhìn thấy" thông tin xa đến đâu?**
Sau `k` layer, node nhìn thấy thông tin trong vùng k-hop: tất cả node cách nó tối đa `k` bước. Receptive field mở rộng dần theo số layer.

**Vì sao thường phải thêm self-loop?**
Để node có thể dùng features của chính nó khi aggregate. Không có self-loop → node không nhận thông điệp từ chính mình trong message passing.

**Tại sao nhiều mô hình chuẩn hóa adjacency / degree?**
Tránh gradient explode khi node có degree lớn (nhiều láng giềng → tổng lớn). Chuẩn hóa bởi degree `D^{-1/2} A D^{-1/2}` (như GCN) giúp tổng hợp thông tin ổn định hơn.

**"Readout" hoặc "pooling" ở graph-level task là gì?**
Tổng hợp tất cả node embeddings của một graph thành một vector duy nhất đại diện cho cả graph. Ví dụ: sum/mean/max trên tất cả node embeddings.

---

## 3. GCN

**GCN học cái gì?**
Học cách tổng hợp features từ láng giềng qua công thức: `H' = σ(Â H W)`, trong đó `Â = D^{-1/2}(A+I)D^{-1/2}` là adjacency matrix đã chuẩn hóa có self-loop, `W` là weights trainable.

**Một lớp GCN làm gì về trực giác?**
"Trộn" features của mỗi node với features của láng giềng theo tỉ lệ nghịch với căn bậc hai tích degree. Mỗi node lấy trung bình có trọng số của neighborhood features rồi áp linear transformation.

**Vì sao GCN thường được mô tả là "trộn" thông tin hàng xóm?**
Phép nhân `Â H` tương đương áp một bộ lọc làm trơn (smoothing filter) trên graph — mỗi node lấy feature là tổng hợp từ chính nó và láng giềng, làm features gần nhau trên graph trở nên gần nhau hơn.

**Vì sao GCN thường hợp với semi-supervised node classification?**
GCN có thể học trên toàn bộ graph structure ngay cả khi chỉ một số node có nhãn — truyền thông tin nhãn qua cấu trúc đồ thị. Semi-supervised setup tự nhiên với transductive GNN.

**GCN là spectral hay spatial?**
Về mặt lý thuyết có gốc spectral (xấp xỉ bậc nhất của spectral filter), nhưng thực ra có thể hiểu và implement như spatial (message passing). Thường được nhắc là cầu nối giữa hai quan điểm.

**Vì sao GCN thường chỉ cần 2-3 layers?**
Mỗi layer mở rộng receptive field thêm 1-hop. Với graph mạng xã hội (diameter nhỏ, dense), 2-3 hop là đủ. Thêm layer → oversmoothing.

**Vì sao GCN dễ bị giảm chất lượng khi tăng depth?**
Sau nhiều layer, embeddings của tất cả nodes hội tụ về cùng một vector (oversmoothing) — mất khả năng phân biệt giữa các node.

---

## 4. GraphSAGE

**GraphSAGE khác GCN ở điểm nào?**
GCN: học trực tiếp embedding cho từng node (transductive, cần toàn bộ graph). GraphSAGE: học hàm tổng hợp từ mẫu neighbor → có thể generate embedding cho node chưa thấy (inductive).

**Vì sao GraphSAGE được gọi là inductive?**
Học một hàm tổng hợp (`f(neighborhood)`) thay vì lookup embedding → có thể áp dụng cho node mới chỉ cần biết features và neighbors của nó, không cần retrain.

**"Sample and aggregate" nghĩa là gì?**
Thay vì dùng toàn bộ láng giềng (có thể rất nhiều), GraphSAGE lấy mẫu ngẫu nhiên một số cố định láng giềng → tính toán scalable, batch training trên graph lớn.

**Vì sao GraphSAGE hữu ích khi graph lớn?**
Neighbor sampling kiểm soát kích thước neighborhood tại mỗi bước → mini-batch training khả thi. GCN cần toàn bộ adjacency matrix → không fit vào memory với graph rất lớn.

**GraphSAGE xử lý node chưa từng thấy lúc train như thế nào?**
Chỉ cần features và neighbor features của node đó → feed vào aggregation function đã học → có embedding ngay lập tức, không cần retrain.

**So sánh mean aggregator, LSTM aggregator, pooling aggregator trong GraphSAGE:**
Mean: simple, hiệu quả. LSTM: xử lý neighbors theo thứ tự ngẫu nhiên — không tự nhiên permutation-invariant, nhưng expressive hơn. Pooling (max): áp MLP lên từng neighbor rồi max-pool — học được transformation tốt hơn mean đơn giản.

---

## 5. GAT

**GAT khác GCN ở chỗ nào?**
GCN: trọng số aggregate cố định theo degree (symmetric normalization). GAT: trọng số aggregate được học qua attention mechanism — mỗi láng giềng có importance khác nhau.

**Attention coefficient trong GAT có ý nghĩa gì?**
`α_{ij}` là mức độ "quan trọng" của node `j` khi cập nhật embedding node `i`. Được tính từ features của cả hai node qua một shared attention mechanism.

**Vì sao attention giúp phân biệt mức quan trọng của hàng xóm?**
Cho phép model học "node nào trong neighborhood đáng tin hơn" dựa trên features — thay vì đối xử bằng nhau hoặc chỉ theo degree.

**Multi-head attention trong GAT để làm gì?**
Chạy `K` attention mechanisms song song, concat hoặc average kết quả → stable training, học nhiều loại "quan trọng" khác nhau, tương tự Transformer.

**GAT có lợi gì so với việc gán trọng số đều như GCN?**
Linh hoạt hơn: model tự học phải chú ý vào láng giềng nào. Phù hợp khi mức quan trọng của mỗi láng giềng khác nhau rõ ràng.

**GAT có nhược điểm gì khi graph rất lớn?**
Attention cần tính toán cho mỗi cặp (node, neighbor) → chi phí tăng với degree lớn. Không scalable bằng GraphSAGE với sampling.

---

## 6. GIN, expressivity và WL test

**"Expressive power" của GNN nghĩa là gì?**
Khả năng phân biệt hai graph (hoặc hai node) khác nhau về cấu trúc. GNN mạnh hơn = có thể phân biệt nhiều cặp graph khác nhau hơn.

**Vì sao nhiều GNN không phân biệt được một số graph khác nhau?**
Vì khi hai graph có cùng neighborhood structure (theo góc nhìn WL), mean/max aggregation sẽ tạo ra cùng embedding — mặc dù graph thực sự khác nhau.

**GIN được đề xuất để giải quyết hạn chế gì?**
Tối đa hóa expressive power trong lớp neighborhood aggregation GNN — đạt sức biểu đạt ngang bằng Weisfeiler-Lehman (WL) isomorphism test bậc 1.

**GIN liên hệ thế nào với Weisfeiler-Lehman test?**
WL test lặp lại: hash multiset của neighbor labels để cập nhật label cho mỗi node. GIN mô phỏng điều này: dùng sum aggregation (phân biệt được multisets tốt nhất) + MLP đủ mạnh để "hash".

**Vì sao sum aggregation thường mạnh hơn mean/max về mặt phân biệt cấu trúc?**
Sum giữ thông tin về số lượng láng giềng. Mean/max mất thông tin đó: `{a,a,b}` và `{a,b}` có mean khác nhau nhưng sum khác nhau rõ ràng hơn trong nhiều trường hợp.

---

## 7. R-GCN và graph quan hệ

**Khi nào graph là multi-relational?**
Khi có nhiều loại quan hệ giữa các node: trong knowledge graph, node là entity, edge có nhiều loại (là-cha-của, làm-việc-tại, nằm-ở...).

**R-GCN khác GCN chuẩn ở điểm nào?**
Mỗi loại quan hệ `r` có weight matrix riêng `W_r`. Message passing tổng hợp riêng theo từng relation type rồi kết hợp: `h_v' = σ(Σ_r Σ_{u∈N_r(v)} W_r h_u / |N_r(v)| + W_0 h_v)`.

**Vì sao knowledge graph hay dùng R-GCN?**
KG có nhiều loại relation khác nhau (hàng trăm relation types) → cần model phân biệt loại quan hệ khi aggregate. R-GCN xử lý self nhiều relation type tự nhiên.

**Relation type được đưa vào message passing như thế nào?**
Mỗi edge gắn với một relation type `r`. Khi tổng hợp thông điệp cho node `v`, group các edge theo type và dùng matrix tương ứng `W_r` cho từng group.

**R-GCN thường dùng cho node classification hay link prediction?**
Cả hai. Node classification: classify entity type. Link prediction (knowledge graph completion): dự đoán missing relation giữa hai entity.

---

## 8. Training và thực hành

**Loss cho node classification thường là gì?**
Cross-entropy trên các node có nhãn: `L = -Σ_{v có nhãn} Σ_c y_{vc} log(ŷ_{vc})`.

**Loss cho link prediction thường xây thế nào?**
Binary cross-entropy: positive pairs (edge có thật) vs negative pairs (edge không tồn tại). Hoặc BPR loss: so sánh score positive edge > negative edge.

**Vì sao link prediction hay cần negative sampling?**
Số edge không tồn tại (potential edges) rất lớn so với số edge thật. Phải sample một tập negative edges đủ nhỏ và đủ khó để train hiệu quả.

**Với graph classification, vì sao phải có bước pooling/readout?**
Mỗi graph có số node khác nhau và size khác nhau → không thể trực tiếp dùng tập node embeddings. Readout tổng hợp thành một vector cố định đại diện cả graph.

**Mini-batch cho GNN khác mini-batch trong CNN thế nào?**
CNN: sample ảnh ngẫu nhiên, independent. GNN: node embeddings phụ thuộc nhau qua neighborhood → phải include cả subgraph (neighbors, neighbors-of-neighbors...) vào batch — "computation graph" cho mỗi node.

**Vì sao train GNN trên graph lớn cần neighbor sampling hoặc subgraph sampling?**
Nếu lấy toàn bộ neighborhood cho mỗi node, computation graph có thể bùng nổ (exponential theo depth) → không fit vào GPU. Sampling giới hạn số láng giềng được xem xét.

**Dữ liệu train/val/test của graph có gì dễ bị leakage?**
Node/edge ở test set kết nối với node ở train set → test nodes "thấy" training information qua message passing. Cần thiết kế split cẩn thận (time-based split, inductive split).

**Khi nào split theo node, theo edge, hay theo graph?**
Node split: khi bài toán là node classification trên một graph. Edge split: link prediction. Graph split: graph-level task với nhiều graphs độc lập.

---

## 9. Hạn chế của GNN

**Oversmoothing là gì?**
Sau quá nhiều message passing layers, embedding của tất cả node trở nên quá giống nhau — mất khả năng phân biệt. Nguyên nhân: mỗi layer là một bước làm trơn (averaging) trên graph.

**Vì sao thêm nhiều layer làm embedding các node ngày càng giống nhau?**
Mỗi layer trộn embedding của node với láng giềng. Sau `k` layer, mỗi node "thấy" tất cả node trong k-hop neighborhood → các node trong cùng component ngày càng hội tụ về một vector.

**Oversquashing là gì?**
Quá nhiều thông tin từ xa bị nén vào vector cố định kích thước nhỏ qua các nút cổ chai (bottleneck edges) của graph. Thông tin bị "bóp méo" — mô hình khó truyền thông tin từ node xa đến node đích.

**Oversmoothing khác oversquashing ở đâu?**
Oversmoothing: embeddings quá giống nhau (representation collapse). Oversquashing: thông tin bị mất khi truyền qua nhiều bước qua cấu trúc graph hẹp — ngay cả khi depth chưa đủ gây oversmoothing.

**Vì sao GNN hay gặp khó với long-range dependencies?**
Phải có nhiều layer để thông tin di chuyển từ xa. Nhưng nhiều layer → oversmoothing. Hai vấn đề mâu thuẫn: cần depth để độ dài, nhưng depth gây collapse.

**Skip connection hoặc residual connection giúp gì cho GNN?**
Như ResNet: giữ thông tin ban đầu của node không bị "trộn" hoàn toàn. Giúp chống oversmoothing và cho phép train mạng sâu hơn.

---

## 10. Phân biệt

**GCN vs GraphSAGE:** GCN transductive (cần toàn bộ graph khi train), fixed aggregation. GraphSAGE inductive (học aggregation function), neighbor sampling, xử lý được node mới.

**GCN vs GAT:** GCN: trọng số aggregate cố định theo degree. GAT: attention có thể học — linh hoạt hơn nhưng thường chậm hơn.

**GraphSAGE vs GAT:** GraphSAGE: scalable, sampling tốt. GAT: attention học được importance. Có thể kết hợp: GAT với neighbor sampling.

**Node classification vs graph classification:** Node: gán nhãn từng node trong một graph lớn. Graph: gán nhãn cả graph — cần readout pooling thêm.

**Transductive vs inductive:** Transductive: test nodes đã thấy trong training (nhưng không có nhãn). Inductive: test trên graph hoàn toàn mới.

**Mean vs sum aggregation:** Mean: không nhạy với kích thước neighborhood. Sum: nhạy với số láng giềng — expressive hơn về mặt lý thuyết (GIN).

**Message passing vs attention-based aggregation:** Message passing chuẩn: trọng số cố định hoặc theo degree. Attention: trọng số học được → phân biệt importance.

**Oversmoothing vs oversquashing:** Oversmoothing: tất cả nodes giống nhau sau nhiều layers. Oversquashing: thông tin far-away bị mất qua bottleneck.

---

## 11. Câu tự luận

**Trình bày pipeline của một mô hình message passing GNN:**
1. Khởi tạo node embedding từ node features.
2. Lặp qua `K` layers: (a) Với mỗi node, tính message từ các neighbor; (b) Aggregate messages (sum/mean/max/attention); (c) Update embedding qua MLP.
3. Sau `K` layers: dùng final embedding cho task (node classification: softmax; link prediction: dot product; graph: readout → MLP).

**Vì sao GNN phù hợp với dữ liệu graph hơn MLP/CNN thường?**
MLP/CNN không biết cấu trúc kết nối của graph — xử lý mọi node như độc lập, phải tạo thủ công feature của neighborhood. GNN tự nhiên truyền thông tin qua cạnh → tận dụng cấu trúc topology, học được graph-aware representation.

**So sánh GCN, GraphSAGE, GAT:**
| | GCN | GraphSAGE | GAT |
|---|---|---|---|
| Aggregation | Normalized sum | Mean/LSTM/Pooling | Attention |
| Inductive | Không | Có | Có |
| Scalability | Kém (full graph) | Tốt (sampling) | Trung bình |
| Khi dùng | Small graph, semi-supervised | Large graph, node mới | Khi cần chú ý láng giềng theo importance |

**Vì sao GIN được xem là mạnh hơn nhiều GNN phổ biến về expressivity?**
Chứng minh được GIN với sum aggregation + MLP đủ mạnh là upper bound của lớp GNN neighborhood aggregation — đạt sức biểu đạt tối đa như WL test bậc 1. GCN/GraphSAGE với mean aggregation yếu hơn vì không phân biệt được một số multiset láng giềng khác nhau.

**Tại sao GNN sâu thường khó train?**
Oversmoothing: nhiều layer → embeddings hội tụ → không phân biệt được. Oversquashing: thông tin xa bị nén mất. Gradient vanish qua nhiều bước message passing. Thường 2-4 layer là đủ và tốt nhất cho hầu hết graph tasks.

**Nếu làm bài toán knowledge graph / molecule / social network, chọn biến thể GNN nào?**
- Knowledge graph: R-GCN (nhiều relation types), hoặc RGAT (có attention theo relation).
- Molecule: GIN hoặc MPNN — cần expressivity cao để phân biệt cấu trúc phân tử tương tự.
- Social network lớn: GraphSAGE (inductive, scalable với node mới, neighbor sampling).
