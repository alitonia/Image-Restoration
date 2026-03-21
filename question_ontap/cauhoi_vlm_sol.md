# Giải Đáp Câu Hỏi Vision-Language Models

## 1. Câu nền tảng

**VLM là gì?**
Mô hình kết hợp hiểu ảnh và ngôn ngữ trong cùng một hệ thống — có thể xử lý input đa phương thức (ảnh + text) và sinh text output liên quan đến ảnh.

**VLM khác LLM thuần văn bản ở đâu?**
LLM chỉ xử lý text. VLM có thêm vision encoder xử lý ảnh và cơ chế kết nối hai modality — có thể trả lời câu hỏi về ảnh, mô tả ảnh, lý luận từ ảnh.

**Image-text alignment là gì?**
Quá trình học sao cho embedding của ảnh và embedding của text mô tả cùng một nội dung "gần nhau" trong không gian vector chung. Nền tảng của VLM.

**Vision encoder và language model mỗi phần làm gì?**
Vision encoder (thường ViT hoặc CNN): trích xuất visual features từ ảnh → patch/region embeddings. Language model: xử lý text và visual tokens để sinh output.

**VLM khác pipeline "CV model + rules + LLM" ở điểm nào?**
Pipeline ghép rời: thông tin mất mát qua các bước, không end-to-end, khó generalize. VLM: học aligned representation end-to-end → thông tin visual và textual tương tác trực tiếp.

---

## 2. CLIP và contrastive alignment

**CLIP học cái gì?**
Học hai encoder (image và text) sao cho embedding của ảnh và caption tương ứng gần nhau trong một không gian chung, embedding của ảnh và caption không tương ứng thì xa nhau. Objective: contrastive loss trên N cặp ảnh-text trong batch.

**Contrastive learning trong CLIP hoạt động ra sao?**
Với batch N cặp ảnh-text: tính N×N similarity matrix. Tối đa similarity trên N cặp đúng (diagonal), tối thiểu trên N²-N cặp sai. InfoNCE loss.

**Zero-shot classification bằng CLIP hoạt động như thế nào?**
Với mỗi class, tạo text prompt: "a photo of a [class]". Encode tất cả prompts. Encode ảnh test. So sánh cosine similarity ảnh vs mỗi class prompt → lấy class có similarity cao nhất. Không cần fine-tune.

**CLIP mạnh ở điểm nào so với supervised classifier truyền thống?**
Zero-shot: không cần labeled data cho class mới, chỉ cần text description. Flexible: class mới thêm vào dễ dàng qua text. Pretrain trên 400M pairs → generalize rộng.

**Vì sao CLIP đặc biệt hữu ích cho retrieval?**
Cùng embedding space → tìm ảnh từ text query (hoặc ngược lại) bằng nearest neighbor search trong embedding space.

**Hạn chế chính của CLIP nếu so với LLaVA/Flamingo?**
CLIP chỉ produces similarity scores — không sinh text tự do. Không trả lời câu hỏi mở, không giải thích, không multi-turn dialogue. Là alignment model, không phải generative assistant.

---

## 3. Kiến trúc VLM

**Kiến trúc VLM chuẩn gồm những khối nào?**
1. Vision encoder: ảnh → visual features/tokens. 2. Connector/bridge: map visual features sang LLM input space. 3. LLM: xử lý visual tokens + text tokens → sinh text output.

**Vì sao phải có "connector" giữa vision encoder và LLM?**
Vision encoder và LLM được train riêng trên domain khác nhau → different embedding spaces → "modality gap". Connector học cách map visual features sang không gian LLM có thể hiểu.

**Projector, adapter, cross-attention, Q-Former khác nhau thế nào?**
Projector (linear/MLP): đơn giản nhất — linear map visual features → LLM tokens. Cross-attention: LLM attend vào visual features qua cross-attention layers. Q-Former: một transformer nhỏ có learnable query vectors "lọc" thông tin từ visual features → compact visual tokens.

**BLIP-2 dùng Q-Former để làm gì?**
Q-Former là lightweight Transformer với fixed số lượng learnable query tokens. Attend vào frozen image encoder → trích xuất compressed visual representation → feed vào frozen LLM. Chỉ Q-Former được train → chi phí thấp.

**Vì sao frozen vision encoder + frozen LLM hấp dẫn về compute?**
Tận dụng pretrained knowledge của cả hai mà chỉ cần train phần connector nhỏ → tiết kiệm compute và tránh catastrophic forgetting.

**End-to-end train và train connector-only khác nhau ra sao?**
Connector-only: nhanh, ổn định, tận dụng backbones. Nhưng visual và language representations không adapt cùng nhau → modality gap có thể không đóng hoàn toàn. End-to-end: tốn hơn nhưng tất cả components học cùng nhau → alignment sâu hơn.

**Khi nào chỉ cần linear/MLP projector là đủ?**
Khi vision encoder và LLM đã được pretrain tốt, task không quá phức tạp (captioning cơ bản, VQA đơn giản), và muốn kiến trúc đơn giản như LLaVA.

**Khi nào cần connector mạnh hơn kiểu cross-attention/Q-Former?**
Khi muốn compress nhiều visual features xuống ít tokens (tiết kiệm LLM context), khi vision và language domain rất khác nhau, hoặc khi cần flexible integration như Flamingo.

---

## 4. Flamingo và interleaved multimodal prompting

**Flamingo giải bài toán gì?**
VLM hỗ trợ chuỗi xen kẽ ảnh và văn bản làm input, few-shot learning với ví dụ multimodal trong prompt, và trả lời câu hỏi mở về ảnh/video.

**"Interleaved image-text inputs" nghĩa là gì?**
Input có thể là: "Đây là ảnh 1: [img1]. Câu hỏi: ... Đây là ảnh 2: [img2]. Câu hỏi: ..." — ảnh và text xen kẽ nhau trong cùng một prompt.

**Cross-attention trong Flamingo đóng vai trò gì?**
Thêm cross-attention layers vào frozen LLM — decoder attend vào visual features của ảnh liên quan trong context. Cách "cắm" thông tin visual vào LLM mà không thay đổi LLM weights.

**Flamingo khác CLIP ở điểm nào?**
CLIP: alignment model sinh similarity score, không sinh text. Flamingo: generative model sinh text tự do dựa trên cả ảnh và text context, hỗ trợ few-shot prompting multimodal.

**Flamingo khác BLIP-2 ở điểm nào?**
Flamingo: cross-attention layers inserted vào frozen LLM — nặng hơn về compute nhưng xử lý được nhiều ảnh xen kẽ tự nhiên. BLIP-2: Q-Former trích xuất compact visual tokens → feed vào LLM đơn giản hơn.

**Khi nào interleaved prompting đặc biệt hữu ích?**
Khi cần few-shot VLM learning (cho ví dụ ảnh-câu hỏi-trả lời trước khi hỏi ảnh mới), khi task cần reference nhiều ảnh cùng lúc, hoặc khi cần reasoning so sánh nhiều ảnh.

---

## 5. Visual instruction tuning và multimodal chat

**Visual instruction tuning là gì?**
Fine-tune VLM trên dataset gồm các instruction dạng "Nhìn ảnh này và [làm gì đó]" — dạy model biết làm theo hướng dẫn liên quan đến ảnh, không chỉ sinh caption.

**LLaVA khác CLIP ở chỗ nào?**
CLIP: alignment model, chỉ similarity score, không sinh text tự do. LLaVA: generative multimodal assistant — kết hợp CLIP image encoder + LLM (Vicuna/Llama) qua MLP projector, sinh text trả lời về ảnh.

**LLaVA khác BLIP-2 ở chỗ nào?**
BLIP-2: Q-Former phức tạp, multi-stage training, freeze cả encoder lẫn LLM. LLaVA: connector đơn giản (linear MLP), tập trung vào instruction tuning data, train nhanh và đơn giản hơn.

**Vì sao instruction tuning làm VLM "nói chuyện" tự nhiên hơn?**
Dữ liệu instruction-following dạy model format câu trả lời theo đúng kỳ vọng của người dùng (hỏi → trả lời, không chỉ mô tả). Model học "cách" trả lời không chỉ "nội dung".

**Vì sao synthetic instruction data lại hay được dùng?**
Human annotation ảnh-text tốn kém. GPT-4 có thể sinh conversation giả từ caption/bounding box → tạo được data scale lớn với chi phí thấp hơn nhiều.

**Vì sao VLM chat được nhưng vẫn có thể nhìn sai ảnh?**
LLM backbone rất mạnh về text → có thể compensate bằng cách "đoán" từ text prior thay vì nhìn ảnh thật. Perception và reasoning là hai năng lực khác nhau.

---

## 6. Các task thường gặp

**Captioning khác VQA ở đâu?**
Captioning: sinh mô tả tự do cho ảnh — open-ended. VQA: trả lời câu hỏi cụ thể về ảnh — thường có câu hỏi định hướng hơn, đôi khi closed-set (yes/no, chọn đáp án).

**Open-ended VQA và multiple-choice VQA khác nhau thế nào?**
Open-ended: sinh text trả lời tự do — khó evaluate tự động. Multiple-choice: chọn 1 trong N đáp án — dễ evaluate chính xác hơn, nhưng model có thể đoán mà không cần hiểu.

**OCR-heavy VQA khó hơn VQA ảnh tự nhiên ở điểm nào?**
Cần đọc và hiểu text trong ảnh (số, chữ nhỏ, font khác nhau) — vision encoder phải có độ phân giải cao. Nhiều VLM được train chủ yếu trên ảnh tự nhiên → kém với text recognition.

**Bảng, sơ đồ, công thức, map làm VLM khó hơn ảnh tự nhiên vì sao?**
Đòi hỏi đọc được text nhỏ, hiểu structure (row-column của bảng, flow của diagram), biết domain knowledge để interpret đúng. Không thể chỉ dựa vào visual appearance như recognition vật thể thông thường.

---

## 7. Dữ liệu và huấn luyện

**VLM thường được train theo mấy giai đoạn?**
Tiêu biểu 2-3 giai đoạn: (1) Pretraining alignment: học align image-text embedding qua contrastive hoặc captioning. (2) Instruction tuning: dạy follow instructions multimodal. (3) Optional: domain/task adaptation.

**Contrastive objective và generative objective khác nhau thế nào?**
Contrastive (CLIP): maximize similarity cặp đúng, minimize sai → học alignment tốt nhưng không sinh text. Generative (captioning, VQA): train LLM sinh text từ ảnh → hiểu sâu hơn nhưng cần nhiều compute hơn.

**Noisy caption ảnh hưởng gì tới mô hình?**
Web image-text pairs thường noisy (caption không mô tả đúng ảnh). Model học alignment sai → hallucinate nhiều hơn, hiểu ảnh kém hơn. Quality filtering quan trọng.

**Negative sampling quan trọng thế nào trong contrastive pretraining?**
Negative samples phải đủ "khó" — tương tự positive nhưng sai — để model học phân biệt tinh tế. Trong-batch negatives (như CLIP) hiệu quả với batch size lớn.

---

## 8. Benchmark và evaluation

**MMMU đo cái gì?**
Massive Multidisciplinary Multimodal Understanding: ~11.5K câu hỏi, 6 ngành, 30 môn học, đòi hỏi kiến thức chuyên ngành + reasoning multimodal. Đánh giá VLM ở mức "đại học" với nhiều loại visual input: chart, diagram, table, map, music sheet, chemical structure.

**Vì sao benchmark đa phương thức khó hơn text-only?**
Phải evaluate cả visual perception lẫn language reasoning. Nhiều loại visual input (ảnh tự nhiên, OCR, chart, diagram...) mỗi loại yêu cầu năng lực khác. Harder to automate evaluation.

**POPE đánh giá cái gì?**
Polling-based Object Probing Evaluation: đánh giá object hallucination — model có bịa vật thể không có trong ảnh không? Hỏi yes/no về sự tồn tại của vật thể theo 3 strategy (random, popular, adversarial).

**Vì sao object hallucination cần benchmark riêng?**
Hallucination xảy ra ngay cả khi model trả lời fluent và coherent — khó phát hiện qua general VQA accuracy. POPE tập trung chuyên vào faithful grounding.

**VQA accuracy có đủ để kết luận VLM mạnh không?**
Không. VQA accuracy (đặc biệt multiple-choice) không đo hallucination, OCR, reasoning, robustness, hay performance trên domain hẹp. Cần nhiều benchmark bổ sung.

---

## 9. Hallucination và failure modes

**Hallucination trong VLM là gì?**
Model sinh ra thông tin không nhất quán với ảnh — mô tả vật thể, màu sắc, số lượng, text... không có trong ảnh hoặc sai so với ảnh.

**Object hallucination khác factual error ở đâu?**
Object hallucination: bịa vật thể trong ảnh (sai về perception của ảnh cụ thể). Factual error: sai kiến thức về thế giới thật. VLM có thể gặp cả hai loại.

**Vì sao VLM có thể "nhìn" ảnh nhưng vẫn bịa thêm vật thể?**
LLM backbone học từ text — biết "trên bàn thường có cốc, bát, thìa". Khi thấy ảnh bàn ăn, có thể "áp" prior từ text thay vì ground vào ảnh thật.

**Vì sao object co-occurrence trong dữ liệu dễ gây bịa?**
Training data: "cà phê" thường đi kèm "bánh" → model học prior mạnh này. Khi thấy ảnh chỉ có cà phê, vẫn có thể output "bánh" vì prior text quá mạnh.

**Các hướng giảm hallucination phổ biến là gì?**
Hallucination-aware RLHF/DPO (reward model penalize hallucination), constrained decoding, sử dụng retrieval, train với negative examples (ảnh-caption không khớp), POPE-style training signal.

---

## 10. Grounding và reasoning

**Visual grounding là gì?**
Khả năng liên kết text với vùng cụ thể trong ảnh — ví dụ: "con chó màu đen" → bbox của đúng con chó đó trong ảnh.

**Grounding khác alignment thế nào?**
Alignment: học không gian embedding chung giữa ảnh và text ở mức global. Grounding: liên kết text với vùng/vật thể cụ thể trong ảnh — fine-grained và spatial hơn.

**VLM có thật sự "lý luận từ ảnh" hay đôi khi chỉ dùng text prior?**
Thường cả hai. Nghiên cứu chỉ ra nhiều VLM có thể rả lời đúng câu hỏi "multimodal" dựa chủ yếu vào text prior — không cần thật sự nhìn ảnh. Đây gọi là "text bias" trong benchmark.

**Text bias trong benchmark multimodal là gì?**
Câu hỏi có thể trả lời được mà không cần nhìn ảnh — chỉ từ distribution của answers. Ví dụ: "Ảnh có bao nhiêu con chó?" — nếu most ảnh có 1-2 con, model đoán "2" vẫn đúng nhiều. VLM khai thác statistical bias này.

**Vì sao bước từ "recognition" sang "reasoning" là khó nhất trong VLM?**
Recognition: identify objects, text, layout. Reasoning: hiểu relationship, draw conclusions từ visual info, integrate với world knowledge. Reasoning đòi hỏi kết hợp visual perception + abstract thinking + domain knowledge.

---

## 11. Adaptation và fine-tuning

**Vì sao frozen vision encoder hấp dẫn?**
Vision encoder (ViT pretrain) đã rất tốt → không cần update. Tiết kiệm compute và bộ nhớ. Tránh catastrophic forgetting visual knowledge.

**Vì sao frozen LLM hấp dẫn?**
LLM pretrain rất tốn kém. Language capability đã mạnh → chỉ cần dạy thêm "nhìn" mà không phá vỡ "nói". LoRA/QLoRA có thể fine-tune nhẹ nếu cần.

**Khi nào adapter/connector đủ, khi nào không đủ?**
Đủ: task general (captioning, VQA ảnh tự nhiên). Không đủ: domain rất specific (y tế, tài liệu kỹ thuật) — cần fine-tune thêm vision encoder hoặc LLM để adapt domain features.

**Nếu muốn làm VLM cho y khoa/tài liệu kỹ thuật, nên sửa phần nào trước?**
Thường: (1) fine-tune vision encoder với domain data (ảnh CT, X-ray, tài liệu). (2) Instruction-tune LLM với domain Q&A. (3) Optionally train connector với domain image-text pairs.

---

## 12. Phân biệt

**VLM vs LLM:** LLM: text only. VLM: text + image input, có vision encoder, có connector, thường có aligned embedding space.

**CLIP-style alignment vs generative VLM:** CLIP: similarity scores, không sinh text, zero-shot retrieval/classification. Generative (LLaVA, Flamingo): sinh text tự do từ ảnh, multimodal assistant.

**Contrastive pretraining vs instruction tuning:** Contrastive: học alignment image-text embedding. Instruction tuning: dạy model follow instructions liên quan đến ảnh → behavior change.

**CLIP vs Flamingo:** CLIP: alignment model, no text generation. Flamingo: generative, few-shot multimodal, cross-attention để inject visual info vào LLM.

**Flamingo vs BLIP-2:** Flamingo: cross-attention inserted vào LLM, xử lý interleaved image-text. BLIP-2: Q-Former compact visual tokens → simpler LLM integration, more compute-efficient.

**BLIP-2 vs LLaVA:** BLIP-2: phức tạp hơn (Q-Former), multi-stage. LLaVA: đơn giản hơn (MLP projector), focus vào instruction tuning data, dễ reproduce.

**Retrieval vs captioning:** Retrieval: rank/find matching items — metric là recall/AUC. Captioning: sinh mô tả text — metric là BLEU/CIDEr/human eval.

**Alignment vs grounding:** Alignment: global image-text embedding gần nhau. Grounding: link text phrase vào specific region trong ảnh — spatial, fine-grained hơn.

**Perception error vs hallucination:** Perception error: model không nhận ra đúng vật thể trong ảnh (visual capability). Hallucination: model nhận ra nhưng bịa thêm thứ không có hoặc sai (grounding failure).

---

## 13. Câu tự luận

**Trình bày pipeline điển hình của một VLM hiện đại:**
1. Vision encoder (frozen ViT-L/14): ảnh → patch embeddings.
2. Connector (MLP projector hoặc Q-Former): map visual → LLM token space.
3. LLM (frozen hoặc LoRA-tuned): nhận visual tokens + text tokens → sinh text output.
Training: (1) pretraining alignment với image-text pairs lớn; (2) instruction tuning với visual instruction data.

**Vì sao CLIP mở đường cho zero-shot vision-language transfer:**
CLIP học aligned embedding space image-text từ 400M internet pairs. Zero-shot: classification = text similarity — "a photo of [class]" vs image embedding. Không cần labeled per-class data. Mở rộng sang task mới chỉ cần viết text description — không cần collect và annotate dữ liệu từ đầu.

**So sánh Flamingo, BLIP-2, LLaVA:**
Flamingo: cross-attention integration, interleaved inputs, few-shot strong, compute-heavy. BLIP-2: Q-Former compress visual info, frozen backbones, lightweight training, tốt cho image captioning/VQA. LLaVA: MLP projector đơn giản, instruction tuning heavy, multimodal assistant tốt, dễ reproduce và scale.

**Vì sao visual instruction tuning tạo ra multimodal assistant tốt hơn alignment thuần:**
Alignment (CLIP): học "ảnh và text này similar" — không biết cách respond theo instruction. Instruction tuning: (instruction, image) → response — học format, cách lý luận về ảnh, cách trả lời câu hỏi phức tạp. Kết quả: model biết "làm gì" với ảnh, không chỉ "hiểu" ảnh gần text nào.

**Trình bày các nguyên nhân chính gây hallucination trong VLM:**
1. Language prior quá mạnh: LLM đoán từ statistical pattern trong text thay vì ground vào ảnh. 2. Object co-occurrence bias: biết "A thường đi kèm B" → bịa B khi thấy A. 3. Weak grounding: visual features không đủ "kéo" LLM, LLM override với text prior. 4. Training data noise: noisy captions → model học sai alignment. 5. Tự tin thái quá: LLM sinh fluent text dù thông tin sai.

**Vì sao MMMU và POPE quan trọng cho VLM hiện đại:**
MMMU: đánh giá reasoning thật sự ở mức đại học với domain knowledge — không thể chỉ pattern match. Phân biệt VLM thật sự hiểu vs chỉ memorize surface patterns. POPE: đánh giá trực tiếp hallucination — điểm yếu cốt lõi của VLM phổ biến nhất. Hai benchmark bổ sung nhau: MMMU đo khả năng reasoning, POPE đo faithfulness.
