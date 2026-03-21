# Giải Đáp Câu Hỏi Large Language Models

## 1. Câu nền tảng

**LLM là gì?**
Mô hình ngôn ngữ có số tham số rất lớn (hàng tỉ), huấn luyện trên văn bản khổng lồ, có thể sinh ngôn ngữ tự nhiên, trả lời câu hỏi, dịch thuật, lập trình, v.v.

**"Large" ám chỉ cái gì?**
Cả ba: số tham số (billions), dữ liệu (trillions tokens), và compute (GPU-hours). Không có ngưỡng cứng — "large" mang tính tương đối.

**LLM khác language model cổ điển ở điểm nào?**
LM cổ điển (n-gram) chỉ xét vài từ trước. LLM dùng Transformer, xử lý context dài, học biểu diễn ngữ nghĩa sâu, có khả năng xuất hiện (emergent abilities) như lập luận, few-shot learning.

**LLM khác foundation model như thế nào?**
Foundation model là thuật ngữ rộng hơn: model lớn pretrain trên data đa dạng, dùng làm nền cho nhiều task. LLM là một loại foundation model chuyên về ngôn ngữ.

**Vì sao phần lớn LLM hiện đại dựa trên Transformer?**
Self-attention cho phép học phụ thuộc dài trong văn bản, huấn luyện song song (không tuần tự như RNN), scale tốt theo số lớp và dữ liệu.

**LLM khác encoder-only model như BERT ở đâu?**
BERT: encoder-only, bidirectional, học biểu diễn (classification, NER). LLM: thường decoder-only, autoregressive, sinh văn bản từ trái sang phải.

**LLM khác seq2seq model như T5 ở đâu?**
T5: encoder-decoder, phù hợp bài toán có input và output rõ ràng. LLM decoder-only xử lý tất cả dưới dạng "tiếp tục văn bản" — linh hoạt hơn với prompting.

**Vì sao decoder-only rất phổ biến cho chat/generation?**
Đơn giản, linh hoạt: mọi task đều là "sinh tiếp văn bản". Không cần kiến trúc encoder riêng — prompt và response đều là cùng một chuỗi.

**"Next-token prediction" là gì?**
Tại mỗi vị trí, mô hình dự đoán token tiếp theo dựa trên tất cả token trước đó. Loss là cross-entropy giữa dự đoán và token thật.

**Vì sao dự đoán token tiếp theo lại sinh ra nhiều năng lực?**
Để dự đoán tốt từ tiếp theo, mô hình phải "hiểu" ngữ nghĩa, ngữ pháp, logic, sự kiện thế giới — buộc học được biểu diễn phong phú.

**Context window là gì?**
Số token tối đa mô hình có thể xử lý trong một lần. Thông tin ngoài context window bị "quên".

**Vì sao context window dài hơn chưa chắc luôn tốt hơn?**
Chi phí attention tăng bậc hai theo context length. Mô hình có thể "mất tập trung" khi context quá dài (lost-in-the-middle). Chất lượng không đảm bảo với thông tin ở xa.

---

## 2. Tokenization

**Tokenization là gì?**
Chuyển đổi văn bản thô thành chuỗi số (token ids) để đưa vào model. Mỗi token ứng với một từ, phần từ (subword), hoặc ký tự.

**Vì sao LLM không làm việc trực tiếp trên "từ"?**
Từ quá nhiều (vocabulary vô hạn), vấn đề OOV, tiếng đa dạng. Subword tokenization cân bằng tốt giữa vocabulary size và độ bao phủ.

**Token khác word, character, subword thế nào?**
Word: nguyên từ — vocabulary lớn, OOV. Character: từng ký tự — quá dài, mất nghĩa. Subword: đơn vị trung gian — vocabulary vừa phải, bao phủ tốt.

**Vì sao subword tokenization phổ biến?**
Xử lý được từ mới (chia thành các phần đã biết), vocabulary manageable (~30k-100k), hiệu quả cho đa ngôn ngữ.

**BPE và unigram LM tokenizer khác nhau thế nào?**
BPE: học ghép các ký tự/subword thường xuyên cùng xuất hiện theo bottom-up. Unigram: bắt đầu từ vocabulary lớn rồi loại bớt token ít hữu ích, chọn phân tách tối ưu theo xác suất.

**Vì sao cùng một câu có thể ra số token rất khác nhau giữa các tokenizer?**
Mỗi tokenizer có vocabulary và thuật toán phân tách riêng → cùng chữ nhưng ngắt khác nhau → số token khác nhau.

**Tokenizer ảnh hưởng chi phí inference như thế nào?**
Nhiều token hơn → context dài hơn → tốn compute hơn → chậm hơn và đắt hơn. Tokenizer hiệu quả giúp giảm token count cùng nội dung.

**Vì sao tokenization quan trọng đặc biệt với đa ngôn ngữ?**
Tiếng Latinh tokenize hiệu quả hơn tiếng Á vì subword matching tốt hơn. Tiếng có ít training data dễ bị "underrepresented" → phải chia nhỏ nhiều → tốn nhiều token hơn.

**OOV problem là gì, và subword xử lý nó ra sao?**
OOV (Out-Of-Vocabulary): từ không có trong vocabulary. Subword chia từ lạ thành các phần nhỏ đã biết → không bao giờ thực sự OOV.

**Vocabulary size ảnh hưởng gì tới model?**
Nhỏ: nhiều token hơn cho cùng text, nhưng embedding nhỏ hơn. Lớn: ít token hơn nhưng embedding matrix to, cần nhiều data để học tốt mỗi token.

**Khi nào tokenizer kém làm hỏng hiệu quả của cả mô hình?**
Khi tokenizer phân tách từ bất hợp lý → model học trên chuỗi token lộn xộn → khó học ngữ nghĩa và ngữ pháp đúng. Đặc biệt tệ với ngôn ngữ không phải tiếng Anh.

---

## 3. Kiến trúc LLM

**Vì sao đa số LLM hiện đại là decoder-only Transformer?**
Đơn giản, linh hoạt, scale tốt. Causal attention phù hợp với autoregressive generation. Mọi task đều quy về "sinh tiếp text".

**Một block của decoder-only Transformer gồm những gì?**
Causal self-attention → Add & LayerNorm → Feed-forward network → Add & LayerNorm. Lặp N lần.

**Causal self-attention là gì?**
Self-attention nhưng mỗi token chỉ attend đến các token trước đó (và chính nó). Mask tam giác trên ngăn "nhìn tương lai".

**Vì sao causal mask là bắt buộc trong autoregressive LM?**
Khi train, toàn bộ chuỗi được xử lý song song. Nếu không mask, mỗi token tại vị trí `t` sẽ thấy token ở `t+1, t+2...` — "gian lận" → model không cần học gì.

**KV cache là gì?**
Trong inference, key và value của các token đã xử lý được lưu lại. Khi sinh token mới, chỉ cần tính Q của token mới rồi attend vào K/V đã cache → không tính lại toàn bộ.

**Vì sao KV cache làm inference nhanh hơn?**
Không cần recompute K,V của toàn bộ context khi sinh mỗi token mới — tiết kiệm rất nhiều compute, tăng throughput.

**Context length tăng thì chi phí attention tăng ra sao?**
Chi phí attention là `O(n²)` theo chiều dài context `n`. Tăng context 2x → 4x compute cho attention.

**Vì sao long-context inference đắt?**
KV cache tăng theo context length → tốn bộ nhớ GPU nhiều. Prefill phase phải xử lý toàn bộ prompt dài → chậm.

**Vì sao batching ở inference khó hơn khi train?**
Các request có độ dài khác nhau, sinh ở tốc độ khác nhau (decode từng token), KV cache per-request → khó batch hiệu quả. Cần continuous batching, paged attention...

---

## 4. Pretraining objective

**Pretraining của LLM là gì?**
Huấn luyện mô hình trên lượng text khổng lồ bằng cách dự đoán token tiếp theo — không cần nhãn thủ công, tự supervised hoàn toàn.

**Cross-entropy loss trong language modeling đo cái gì?**
Đo mức độ "ngạc nhiên" của mô hình trước token đúng: `-log p(x_t | x_{<t})`. Loss thấp = mô hình dự đoán tốt token tiếp theo.

**Perplexity là gì?**
`PPL = exp(average cross-entropy loss)`. Đo mức độ "bối rối" của mô hình. PPL thấp = mô hình dự đoán tốt. PPL=10 nghĩa là mô hình trung bình phân vân giữa 10 lựa chọn đều nhau.

**Vì sao perplexity thấp chưa chắc đồng nghĩa chatbot tốt hơn?**
Perplexity đo khả năng mô hình hóa phân phối văn bản, không đo khả năng trả lời hữu ích, an toàn, hay đúng thực tế — những yếu tố cần instruction tuning và RLHF.

**Causal LM khác masked LM ở đâu?**
Causal (GPT): dự đoán từ trái sang phải — attention một chiều, phù hợp generation. Masked (BERT): che ngẫu nhiên token, dự đoán từ cả hai phía — bidirectional, phù hợp hiểu ngôn ngữ.

**Vì sao MLM mạnh cho hiểu ngôn ngữ nhưng không tự nhiên cho sinh văn bản?**
MLM thấy context hai chiều → học biểu diễn tốt. Nhưng khi sinh, không có token tương lai → không áp dụng tự nhiên cho generation.

---

## 5. Scaling laws và emergent behavior

**Scaling law là gì?**
Mối quan hệ power-law giữa loss mô hình và: số tham số, kích thước data, lượng compute. Tăng một trong ba → loss giảm theo công thức dự đoán được.

**Data scaling, model scaling, compute scaling khác nhau thế nào?**
Data: tăng token training. Model: tăng số tham số. Compute: tăng tổng FLOP bằng cách kết hợp cả hai. Chinchilla law: với compute cố định, nên scale data và model song song (~20 tokens/param là optimal).

**"Emergent abilities" thường được hiểu là gì?**
Năng lực xuất hiện đột ngột khi model đạt quy mô đủ lớn — không có ở model nhỏ, nhưng bột phát ở ngưỡng nào đó. Ví dụ: arithmetic, chain-of-thought reasoning.

**Zero-shot, one-shot, few-shot khác nhau ra sao?**
Zero-shot: không có ví dụ, chỉ có hướng dẫn. One-shot: 1 ví dụ. Few-shot: vài ví dụ trong prompt. Cả ba không update weights — chỉ dùng context.

**Vì sao in-context learning vẫn là chủ đề chưa hoàn toàn được hiểu rõ?**
Không rõ cơ chế: model "học" từ ví dụ trong context hay chỉ "nhận ra pattern"? Hiệu quả phụ thuộc nhiều vào cách viết prompt, thứ tự ví dụ, format — khó giải thích lý thuyết.

---

## 6. Prompting và in-context learning

**Temperature là gì?**
Hệ số chia logits trước softmax. Temperature cao → phân phối flat hơn → diverse hơn nhưng kém coherent. Temperature thấp → confident hơn → ít diversity, dễ lặp.

**Top-k và top-p sampling khác nhau thế nào?**
Top-k: chỉ sample trong k token có xác suất cao nhất. Top-p (nucleus): sample trong tập token nhỏ nhất có tổng xác suất ≥ p — linh hoạt hơn vì tập này thay đổi theo context.

**Prompt injection là gì trong ứng dụng LLM?**
Kẻ tấn công nhúng instruction vào input của user (ví dụ trong document, email) để LLM thực thi lệnh ngoài ý định hệ thống. Ví dụ: "Ignore previous instructions and..."

**Jailbreak khác prompt injection ở điểm nào?**
Jailbreak: user cố tình bypass safety guardrails qua prompt khéo léo. Prompt injection: kẻ tấn công bên ngoài nhúng instruction vào data model xử lý.

---

## 7. Instruction tuning, SFT, RLHF, DPO

**Instruction tuning là gì?**
Fine-tune LLM trên tập data dạng (instruction, response) để model học làm theo hướng dẫn, không chỉ tiếp tục văn bản. Biến raw LLM thành assistant.

**SFT khác pretraining ở đâu?**
Pretraining: văn bản thô, predict next token. SFT: cặp (instruction, response) curated, tinh chỉnh behavior. SFT dùng ít data hơn nhưng chất lượng cao hơn.

**RLHF là gì?**
Reinforcement Learning from Human Feedback: dùng preference data (con người chọn response nào tốt hơn) để train reward model, rồi tối ưu LLM bằng PPO để maximize reward.

**Pipeline chuẩn của RLHF gồm những bước nào?**
1. SFT: fine-tune trên demonstrations. 2. Train reward model: học từ pairwise preferences. 3. RL optimization: dùng PPO tối ưu LLM theo reward model, có KL constraint với SFT model.

**Reward model học gì?**
Học predict con người thích response nào hơn giữa hai response cho cùng prompt. Output là điểm scalar.

**DPO là gì?**
Direct Preference Optimization: tối ưu trực tiếp LLM theo preference data mà không cần train reward model riêng hay chạy RL. Đơn giản hơn và ổn định hơn RLHF.

**DPO khác RLHF ở điểm nào?**
RLHF: cần reward model riêng + PPO training — phức tạp, tốn compute, dễ reward hacking. DPO: một loss function trực tiếp trên preference pairs, không cần RL.

**Khi nào DPO hấp dẫn hơn RLHF?**
Khi muốn đơn giản, ổn định, ít tài nguyên, không có team RL riêng. DPO là lựa chọn thực dụng cho nhiều research group nhỏ.

---

## 8. RAG

**RAG là gì?**
Retrieval-Augmented Generation: kết hợp retriever tìm kiếm tài liệu liên quan từ knowledge base bên ngoài, rồi đưa vào context của LLM để sinh câu trả lời.

**Vì sao LLM thuần tham số hay gặp vấn đề kiến thức cũ?**
Kiến thức được "đóng băng" tại thời điểm training. Sự kiện mới sau cutoff date không được biết → cần RAG để inject kiến thức cập nhật.

**Retriever và generator mỗi phần làm gì?**
Retriever: tìm k đoạn văn bản liên quan nhất từ corpus (dùng embedding similarity). Generator: LLM nhận prompt + retrieved docs → sinh câu trả lời dựa trên context đó.

**Chunking ảnh hưởng chất lượng RAG ra sao?**
Chunk quá nhỏ: mất context xung quanh. Chunk quá lớn: nhiễu, recall giảm, tốn context window. Cần chunk có kích thước hợp lý và có overlap.

**Vì sao RAG có thể giảm hallucination nhưng không loại bỏ hoàn toàn?**
LLM vẫn có thể bỏ qua retrieved context, pha trộn thông tin sai từ parametric memory, hoặc retrieved doc bản thân đã sai/lỗi thời.

**Khi nào nên dùng RAG thay vì fine-tune?**
Khi kiến thức thay đổi liên tục, khi cần cite nguồn, khi domain-specific data nhiều và không muốn retrain. RAG rẻ hơn fine-tune để update knowledge.

**Khi nào nên fine-tune thay vì RAG?**
Khi muốn thay đổi style/behavior của model, khi task có pattern cụ thể cần học (format, cách trả lời), khi độ trễ retrieval là vấn đề.

---

## 9. PEFT: LoRA và QLoRA

**LoRA là gì?**
Low-Rank Adaptation: thay vì fine-tune toàn bộ weight matrix `W`, thêm hai ma trận nhỏ `A` và `B` (rank thấp) với `ΔW = AB`. Chỉ train `A, B` — freeze `W` gốc.

**"Low-rank adaptation" có ý nghĩa trực giác gì?**
Giả định rằng sự thay đổi cần thiết của weight khi fine-tune chỉ nằm trong một không gian con low-rank. Thay vì học toàn bộ `d×d` ma trận, chỉ học `d×r + r×d` với `r << d`.

**Vì sao LoRA tiết kiệm bộ nhớ?**
Chỉ lưu gradient và optimizer state cho `A, B` (rất nhỏ), không phải cho `W` gốc (rất lớn). Có thể giảm trainable params 10-1000x.

**Rank `r` ảnh hưởng thế nào?**
`r` lớn: nhiều tham số hơn, biểu đạt tốt hơn, nhưng tốn bộ nhớ hơn. `r` nhỏ: rẻ hơn nhưng có thể không đủ capacity. `r=4-16` thường là đủ cho nhiều task.

**QLoRA là gì?**
LoRA + quantization: model gốc được load ở 4-bit (NF4 format), chỉ LoRA adapters được train ở precision cao (bf16). Giảm memory rất mạnh — có thể fine-tune 65B model trên 1 GPU.

**4-bit quantization trong QLoRA giúp gì?**
Giảm memory footprint của frozen weights từ 16-bit xuống 4-bit → giảm 4x. Có thể fit model lớn hơn vào GPU ít bộ nhớ hơn.

**Khi nào full fine-tuning vẫn đáng làm hơn LoRA/QLoRA?**
Khi task rất khác với pretraining (domain rất specific), khi có đủ compute và memory, hoặc khi cần đạt performance tối đa mà compute không phải ràng buộc.

---

## 10. Quantization và inference

**Quantization là gì?**
Giảm precision của weights từ float32/bfloat16 xuống int8 hoặc int4 để giảm bộ nhớ và tăng tốc inference. Trade-off: giảm accuracy một chút.

**Vì sao inference LLM thường bị giới hạn bởi memory bandwidth?**
Với batch size nhỏ (chatbot), mỗi token sinh phải load toàn bộ weights từ HBM vào GPU cores. Tốc độ phụ thuộc memory bandwidth, không phải compute. → Memory bandwidth là bottleneck.

**Prefill và decode phase khác nhau ra sao?**
Prefill: xử lý toàn bộ prompt một lần (compute-bound, song song). Decode: sinh từng token một, cần load weights mỗi bước (memory-bound, tuần tự).

**Vì sao token đầu tiên thường chậm hơn các token sau?**
Token đầu tiên cần hoàn tất prefill phase (xử lý toàn bộ prompt). Các token sau chỉ cần decode step nhỏ với KV cache sẵn có.

---

## 11. MoE

**MoE là gì?**
Mixture of Experts: mỗi token chỉ được xử lý bởi một hoặc vài "expert" (sub-network) được router chọn, thay vì toàn bộ FFN. Tăng số tham số tổng nhưng compute mỗi token không tăng tương ứng.

**Dense model và sparse model khác nhau thế nào?**
Dense: mọi tham số đều tham gia xử lý mỗi token. Sparse (MoE): mỗi token chỉ kích hoạt một phần nhỏ expert → cùng compute nhưng nhiều tham số hơn.

**Vì sao load balancing quan trọng trong MoE?**
Nếu tất cả token đều chọn một vài expert → các expert khác không được train → lãng phí. Cần auxiliary loss để đảm bảo phân bố token đều giữa các experts.

**Vì sao MoE hay khó triển khai phân tán hơn dense model?**
Mỗi expert thường nằm ở GPU khác nhau → expert parallelism phức tạp, routing cần all-to-all communication giữa GPUs → latency cao và engineering phức tạp.

---

## 12. Evaluation

**Vì sao không nên chỉ nhìn perplexity?**
Perplexity đo language modeling, không đo instruction following, truthfulness, helpfulness hay safety. Model có PPL thấp vẫn có thể hallucinate hay refuse quá nhiều.

**Benchmark contamination là gì?**
Khi test data của benchmark vô tình xuất hiện trong training data → model "nhớ" đáp án thay vì thật sự hiểu → kết quả benchmark không phản ánh thực lực thật.

**Truthfulness khác factuality ở đâu?**
Factuality: thông tin có đúng thực tế không. Truthfulness: model có cố tình nói đúng theo hiểu biết của nó không (không nói điều biết là sai). Model có thể factually sai nhưng truthful nếu nó tin điều đó là đúng.

**Calibration của LLM nghĩa là gì?**
Model tự tin vào câu trả lời đúng mức với xác suất đúng của nó. Calibrated: khi model nói "80% chắc" thì ~80% lần nó đúng. LLMs thường overconfident.

---

## 13. Hallucination và sai sót

**Hallucination là gì?**
Model sinh ra thông tin trông có vẻ tự tin và flowing nhưng không có cơ sở hoặc sai so với thực tế — "bịa" có vẻ thật.

**Vì sao LLM có thể "nói rất tự tin nhưng sai"?**
Training objective là sinh text fluent — sự tự tin trong ngôn ngữ học từ style của văn bản training, không phải từ kiểm tra factual. Model không "biết" nó không biết.

**Vì sao next-token prediction không đảm bảo truthfulness?**
Model tối ưu fluency và coherence, không tối ưu factual correctness. Web training data chứa cả thông tin sai — model học cả hai pattern.

**Cách giảm hallucination phổ biến là gì?**
RAG (inject sources), RLHF/DPO (reward truthfulness), prompt engineering (yêu cầu cite nguồn, "nếu không biết hãy nói"), constrained decoding, fine-tune với factual QA data.

**Vì sao "từ chối khi không chắc" là một năng lực alignment quan trọng?**
Model nói "Tôi không chắc" khi thực sự không biết an toàn hơn nhiều so với hallucinate tự tin. Calibrated uncertainty là một mục tiêu của alignment.

---

## 14. An toàn và ứng dụng

**Alignment khác safety ở đâu?**
Alignment: làm model hành xử đúng ý định con người (helpful, honest). Safety: ngăn hại cụ thể (không sinh nội dung nguy hiểm, không bị exploit). Safety thường là tập con của alignment.

**Over-refusal là gì?**
Model từ chối quá nhiều — ngay cả những câu hỏi vô hại — vì quá thận trọng với safety. Giảm helpfulness mà không tăng safety thực sự.

**Vì sao guardrails không thể chỉ dựa vào regex/filter đơn giản?**
LLM có thể diễn đạt nội dung harmful theo vô số cách khác nhau — không thể liệt kê hết. Cần model-based filtering hoặc classifier để hiểu ngữ nghĩa.

**Khi nào nên dùng model nhỏ nội bộ thay vì model lớn API?**
Khi data nhạy cảm không thể gửi ra ngoài, khi cần latency thấp, khi muốn kiểm soát hoàn toàn, hoặc khi cost API quá cao với traffic lớn.

---

## 15. Phân biệt

**LLM vs foundation model:** Foundation model rộng hơn (bao gồm VLM, audio model...). LLM là foundation model chuyên ngôn ngữ.

**Decoder-only vs encoder-only:** Decoder: autoregressive, sinh text, causal attention. Encoder: bidirectional, hiểu ngôn ngữ, masked attention.

**Pretraining vs SFT:** Pretraining: web text thô, predict next token, scale lớn. SFT: data instruction-response curated, dạy behavior, scale nhỏ hơn.

**SFT vs RLHF:** SFT dạy từ demonstrations (imitation). RLHF tối ưu theo preference feedback qua reward model + RL — phức tạp hơn nhưng align tốt hơn.

**RLHF vs DPO:** RLHF cần reward model + PPO. DPO tối ưu trực tiếp từ preference pairs không qua RL — đơn giản và ổn định hơn.

**Fine-tuning vs RAG:** Fine-tuning thay đổi weights — cố định knowledge. RAG inject context runtime — dynamic, không cần retrain.

**Full fine-tuning vs LoRA:** Full: update tất cả weights, tốn bộ nhớ nhiều. LoRA: chỉ train rank-r adapters nhỏ, rẻ hơn rất nhiều.

**LoRA vs QLoRA:** QLoRA = LoRA + load base model ở 4-bit. Giảm thêm memory đáng kể cho fine-tune model rất lớn.

**Dense Transformer vs MoE:** Dense: mọi tham số kích hoạt. MoE: chỉ một phần expert kích hoạt — nhiều tham số hơn với cùng compute.

**Perplexity vs human preference:** PPL: metric tự động, đo fluency/prediction. Human preference: đo helpfulness, correctness, safety — không tự động, tốn kém.

**Factuality vs truthfulness:** Factuality: đúng thực tế. Truthfulness: model nói theo đúng hiểu biết của nó (không cố tình lừa).

**Hallucination vs outdated knowledge:** Outdated: thông tin đúng tại thời điểm training nhưng đã thay đổi. Hallucination: thông tin không bao giờ đúng, model tự bịa.

---

## 16. Câu tự luận

**Pipeline xây dựng LLM hiện đại từ pretraining tới alignment:**
1. **Pretraining**: train trên hàng nghìn tỉ token web text với next-token prediction — học kiến thức và ngôn ngữ.
2. **SFT**: fine-tune trên tập instruction-response chất lượng cao — dạy follow instructions.
3. **RLHF/DPO**: tối ưu theo human preference — cải thiện helpfulness, harmlessness, honesty.
4. **Safety evaluation & red-teaming**: kiểm tra và vá các điểm yếu an toàn.

**Vì sao next-token prediction có thể sinh ra năng lực few-shot/in-context learning?**
Trong web text, xuất hiện rất nhiều pattern dạng "bài toán-giải pháp", "ví dụ-giải thích". Model học những pattern này → khi thấy few-shot examples trong context, nhận ra cấu trúc tương tự và tự nhiên "tiếp tục" theo đúng pattern đó.

**So sánh BERT-style và GPT-style pretraining:**
BERT (masked LM): che 15% token, dự đoán từ cả hai phía → học biểu diễn bidirectional tốt, phù hợp NLU tasks. GPT (causal LM): predict token tiếp theo từ trái → học autoregressive generation, phù hợp synthesis và in-context learning.

**Trình bày RLHF và các khó khăn thực tế:**
Bước 1: SFT. Bước 2: human annotators so sánh cặp responses → preference data → train reward model. Bước 3: PPO tối ưu LLM maximize reward có KL penalty tránh deviation xa SFT. Khó khăn: reward hacking (model "bịp" reward model), reward model có thể bị exploit, PPO không ổn định, preference annotation tốn kém và không nhất quán.

**So sánh RLHF và DPO:**
RLHF: cần train reward model riêng, rồi PPO — 2 bước nặng, dễ không ổn định. DPO: tính trực tiếp loss từ preference pairs, không cần reward model hay RL — đơn giản hơn, ổn định hơn, kết quả thường tương đương hoặc tốt hơn.

**Khi nào chọn RAG, khi nào chọn fine-tune?**
RAG: knowledge thay đổi thường xuyên, cần cite nguồn, không muốn retrain. Fine-tune: cần thay đổi style/behavior cố định, task có pattern đặc thù cần học sâu, latency retrieval là vấn đề.

**Vì sao LoRA/QLoRA trở thành chuẩn fine-tuning thực dụng cho LLM:**
Full fine-tune 7B+ model cần nhiều GPU A100 — prohibitive. LoRA: giảm trainable params 99%, chất lượng gần tương đương. QLoRA: thêm 4-bit quantization → fine-tune 65B model trên 1 × 48GB GPU. Democratize fine-tuning cho cộng đồng.

**Các nguyên nhân chính gây hallucination và hướng giảm thiểu:**
Nguyên nhân: training data sai/thiếu, model không "biết nó không biết", training objective không tối ưu truthfulness, decoding greedy bỏ qua uncertainty. Hướng giảm: RAG, RLHF với honest preference, calibration training, retrieval + cite, từ chối khi uncertain, chain-of-thought để kiểm tra lại.
