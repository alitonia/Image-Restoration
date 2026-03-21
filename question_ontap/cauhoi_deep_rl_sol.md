# Giải Đáp Câu Hỏi Deep Reinforcement Learning

## 1. Câu nền tảng

**Reinforcement Learning khác gì so với supervised learning?**
Supervised learning học từ nhãn cho sẵn. RL không có nhãn đúng: agent tự thực hiện hành động, nhận phản hồi từ môi trường (reward), và học từ đó. Không ai nói cho agent biết hành động nào là "đúng" ở mỗi bước.

**Agent, environment, state, observation, action, reward là gì?**
- **Agent**: hệ thống học và ra quyết định.
- **Environment**: mọi thứ bên ngoài agent, phản ứng lại hành động.
- **State**: trạng thái đầy đủ của môi trường.
- **Observation**: những gì agent thực sự nhìn thấy (có thể chỉ là một phần của state).
- **Action**: lựa chọn của agent tại mỗi bước.
- **Reward**: tín hiệu số môi trường trả về để đánh giá hành động vừa thực hiện.

**Episode và trajectory khác nhau thế nào?**
Episode: một lượt chơi từ đầu đến cuối (kết thúc khi đạt trạng thái terminal). Trajectory: chuỗi `(s, a, r, s', a', ...)` ghi lại toàn bộ các bước trong một episode hoặc một đoạn tương tác.

**Return là gì? Phân biệt immediate reward và long-term return.**
Immediate reward `r_t` là phần thưởng nhận được ngay tại bước `t`. Return `G_t` là tổng phần thưởng tích lũy từ `t` trở đi (có thể chiết khấu): `G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...`. Mục tiêu RL là tối đa hóa return, không chỉ reward tức thời.

**Discount factor γ có ý nghĩa gì?**
Kiểm soát mức độ "quan tâm đến tương lai". γ gần 1: agent coi phần thưởng xa gần như tương đương phần thưởng gần. γ gần 0: agent chỉ quan tâm phần thưởng ngay trước mắt. γ nhỏ giúp hội tụ dễ hơn nhưng agent có thể "thiển cận".

**Policy là gì? Phân biệt deterministic và stochastic policy.**
Policy `π` là chiến lược ra quyết định: ánh xạ từ state sang action. Deterministic: `a = π(s)` — mỗi state cho đúng một action. Stochastic: `π(a|s)` là phân phối xác suất trên các action — agent chọn ngẫu nhiên theo phân phối đó.

**Value function V(s) là gì?**
`V^π(s)` là kỳ vọng return khi bắt đầu từ state `s` và theo policy `π`. Đo lường "state này tốt đến đâu" nếu chơi theo chính sách hiện tại.

**Q-function Q(s, a) là gì?**
`Q^π(s, a)` là kỳ vọng return khi ở state `s`, thực hiện action `a`, rồi theo policy `π`. Đo lường "action này tốt đến đâu trong state này".

**Advantage A(s, a) là gì, và vì sao cần nó?**
`A(s, a) = Q(s, a) - V(s)`: đo xem action `a` tốt hơn hay kém hơn giá trị trung bình của state đó. Dùng Advantage thay vì raw return giúp giảm variance trong cập nhật policy vì đã trừ đi "baseline" là `V(s)`.

**Bellman equation nói lên điều gì?**
Giá trị của một state bằng reward hiện tại cộng giá trị chiết khấu của state tiếp theo: `V(s) = E[r + γV(s')]`. Đây là cơ sở của hầu hết các thuật toán RL.

**Vì sao exploration-exploitation là bài toán trung tâm của RL?**
Agent cần khám phá (exploration) để phát hiện hành động tốt hơn, nhưng cũng cần khai thác (exploitation) kiến thức hiện có để thu thập reward. Làm nhiều một bên sẽ thiệt bên còn lại — không có giải pháp tổng quát tối ưu.

**MDP là gì? Markov property là gì?**
MDP (Markov Decision Process) là khung hình thức hóa RL gồm `(S, A, P, R, γ)`. Markov property: state tương lai chỉ phụ thuộc state hiện tại, không phụ thuộc lịch sử trước đó — `P(s'|s, a)` là đủ.

---

## 2. Value-based methods và DQN

**Q-learning học cái gì?**
Học trực tiếp hàm Q `Q(s, a)` bằng cách cập nhật theo Bellman equation: `Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s', a') - Q(s,a)]`. Không cần biết model môi trường.

**DQN khác Q-learning tabular ở điểm nào?**
Q-learning tabular lưu Q(s,a) trong bảng — chỉ dùng được khi không gian state nhỏ. DQN dùng mạng nơ-ron để xấp xỉ hàm Q → áp dụng được cho không gian state liên tục và lớn (như pixel Atari).

**Vì sao DQN cần neural network?**
Không gian state của game Atari (ảnh 84×84) có quá nhiều trạng thái để lưu trong bảng. Mạng nơ-ron tổng quát hóa: nhận ảnh pixel làm input, trả về Q-value cho mỗi action.

**Input và output của DQN là gì?**
Input: vài frame ảnh gần nhất (stack frames để biết chuyển động). Output: vector Q-value, mỗi phần tử ứng với một action hợp lệ.

**Loss của DQN được xây từ Bellman target như thế nào?**
`Loss = (r + γ max_{a'} Q_target(s', a') - Q(s, a))²`. Target `r + γ max Q_target(s', a')` đóng vai "nhãn", còn `Q(s,a)` là dự đoán của network.

**Tại sao target có dạng `r + γ max Q(s', a')`?**
Đây là xấp xỉ giá trị Q thật theo Bellman optimality equation: action tốt nhất ở `s'` sẽ cho giá trị `max_{a'} Q(s', a')`. Cộng với reward tức thời `r` và chiết khấu `γ`.

**Vì sao DQN phù hợp với action space rời rạc?**
Output của network là Q-value cho từng action. Chọn action bằng `argmax` — chỉ hiệu quả khi số action hữu hạn và không quá lớn.

**Vì sao DQN không hợp với continuous action space?**
Không thể `argmax` trên tập action vô hạn. Phải dùng thuật toán khác như DDPG, TD3, SAC.

**Experience replay dùng để làm gì?**
Lưu trữ các transition `(s, a, r, s')` vào buffer, rồi lấy mẫu ngẫu nhiên mini-batch để train. Giúp phá vỡ tính tương quan thời gian giữa các sample liên tiếp và cho phép dùng lại dữ liệu nhiều lần.

**Target network dùng để làm gì?**
Một bản sao của Q-network với weights cập nhật chậm hơn (cố định trong nhiều bước). Dùng để tính Bellman target ổn định hơn, tránh vòng lặp "đuổi mục tiêu đang chạy".

**Hai failure mode phổ biến của DQN là gì?**
1. Q-value overestimation: `max` operator có xu hướng ước lượng Q cao hơn thực tế.
2. Huấn luyện không ổn định: correlation giữa consecutive samples và non-stationary target gây divergence.

**Double DQN ra đời để xử lý vấn đề gì?**
Overestimation: DQN chuẩn dùng cùng network để chọn action lẫn đánh giá — dễ bị bias lên. Double DQN dùng Q-network để chọn action tốt nhất, nhưng dùng target network để đánh giá Q-value của action đó.

**Nếu agent học rất không ổn định trong Atari, kiểm tra gì đầu tiên?**
Learning rate có quá lớn không, replay buffer có đủ lớn không, target network update frequency, reward clipping, và xem Q-value có explode không.

---

## 3. Policy gradient

**Policy gradient học trực tiếp cái gì?**
Học trực tiếp tham số `θ` của policy `π_θ(a|s)`, tối ưu hóa theo hướng gradient của expected return: `∇_θ E[G]`.

**Vì sao policy gradient phù hợp hơn cho stochastic policies?**
Policy gradient tối ưu trực tiếp phân phối xác suất trên actions. Value-based methods lấy `argmax` — vốn deterministic và không tự nhiên với action space liên tục hoặc cần stochasticity.

**Công thức `-log π(a_t|s_t) × R_t` có trực giác nghĩa là gì?**
Nếu return `R_t` cao (episode tốt), loss nhỏ → tăng xác suất action `a_t`. Nếu `R_t` thấp, loss lớn → giảm xác suất action đó. Mạng học "nhiều hành động dẫn đến kết quả tốt, ít hành động dẫn đến kết quả xấu".

**Tại sao return cao làm tăng xác suất lặp lại hành động?**
Gradient cập nhật theo hướng tăng log-probability của action `a_t` khi return cao. Log-probability tăng ↔ xác suất của action đó tăng.

**Vì sao policy gradient thường có variance cao?**
Return `G_t` tích lũy qua nhiều bước, phụ thuộc nhiều yếu tố ngẫu nhiên. Ước lượng từ một episode duy nhất rất nhiễu. Cần nhiều sample để ước lượng gradient chính xác.

**Baseline giúp giảm variance như thế nào?**
Trừ đi một baseline `b(s)` (thường là `V(s)`) khỏi return: dùng `G_t - b(s_t)` thay vì `G_t`. Kỳ vọng gradient không đổi (unbiased), nhưng variance giảm vì baseline tương quan với return, làm "nhiễu" giảm xuống.

**Vì sao advantage thường tốt hơn dùng raw return?**
Advantage `A(s,a) = Q(s,a) - V(s)` biểu diễn "action này so với mức trung bình thế nào", đã loại bỏ phần giá trị chung của state. Raw return bao gồm cả phần tốt/xấu không do action này gây ra.

**On-policy nghĩa là gì? Vanilla policy gradient là on-policy vì sao?**
On-policy: agent chỉ học từ dữ liệu do chính policy hiện tại tạo ra. Policy gradient tối ưu trực tiếp `π_θ` nên phải dùng trajectories sinh bởi `π_θ` — không thể tái sử dụng dữ liệu cũ từ policy khác.

**So sánh policy gradient với DQN:**
DQN: học Q-function, chọn action bằng argmax, off-policy, chỉ hợp với discrete action. Policy gradient: học trực tiếp policy, on-policy, hoạt động với continuous action, có thể stochastic tự nhiên hơn.

**Khi nào nên ưu tiên policy gradient hơn value-based?**
Khi action space liên tục, khi cần stochastic policy tự nhiên, hoặc khi không gian state quá lớn để học Q-function chính xác.

---

## 4. Actor-critic

**Actor là gì? Critic là gì?**
Actor: policy `π_θ(a|s)` — học ra action. Critic: value function `V_φ(s)` hoặc `Q_φ(s,a)` — đánh giá action/state của actor tốt đến đâu.

**Vì sao actor-critic thường ổn định hơn vanilla policy gradient?**
Vanilla policy gradient dùng return thực tế (nhiều noise) để ước lượng gradient. Actor-critic dùng critic ước lượng `V(s)` làm baseline liên tục và chính xác hơn, giảm variance mạnh.

**Critic học V(s), Q(s,a), hay advantage?**
Tùy thuật toán. Thường critic học `V(s)`, rồi advantage được ước lượng từ `r + γV(s') - V(s)`. Một số thuật toán critic học trực tiếp `Q(s,a)`.

**Actor-critic giảm variance bằng cách nào?**
Thay vì dùng return Monte Carlo nhiều noise, dùng critic estimate `r + γV(s') - V(s)` (TD error) để cập nhật actor — ít noise hơn nhiều.

**Điểm đánh đổi giữa bias và variance trong actor-critic là gì?**
Critic estimate có thể không chính xác (biased) nếu critic chưa học tốt. Nhưng variance thấp hơn so với Monte Carlo. Cần cân bằng giữa số bước TD (ít bước → ít variance, nhiều bias; nhiều bước → ngược lại).

**Nếu critic học tệ thì actor bị ảnh hưởng ra sao?**
Actor nhận được gradient không đúng hướng từ critic sai → policy cập nhật theo hướng không tốt → cả hệ thống học sai. Chất lượng critic là nền tảng của toàn hệ thống.

**Tại sao nói actor-critic là "lai" giữa value-based và policy-based?**
Vì có cả hai thành phần: actor tối ưu policy trực tiếp (như policy gradient), critic ước lượng value function (như value-based). Kết hợp ưu điểm của cả hai.

---

## 5. PPO, DDPG, TD3, SAC

**PPO giải quyết vấn đề gì của policy gradient cơ bản?**
Policy gradient cơ bản cập nhật policy với bước có thể quá lớn → policy thay đổi đột ngột → không ổn định. PPO giới hạn mức thay đổi policy ở mỗi bước cập nhật.

**"Clipping" trong PPO có ý nghĩa trực giác gì?**
PPO clip tỷ lệ `π_new/π_old` trong khoảng `[1-ε, 1+ε]`. Nếu policy thay đổi quá nhiều, gradient bị cắt bớt, ngăn không cho bước cập nhật quá lớn phá hỏng policy.

**Vì sao PPO rất phổ biến trong thực hành?**
Đơn giản trong cài đặt, ổn định, sample-efficient hợp lý, hoạt động tốt trên nhiều loại bài toán mà không cần tuning quá nhiều. Dùng được cho cả continuous lẫn discrete action.

**DDPG là gì? Vì sao nó hợp với continuous action?**
DDPG (Deep Deterministic Policy Gradient) kết hợp actor-critic với experience replay. Actor học policy deterministic `a = μ(s)`, critic học Q(s,a). Hợp với continuous action vì actor trả về giá trị liên tục trực tiếp, không cần argmax trên tập vô hạn.

**DDPG là on-policy hay off-policy?**
Off-policy. Dùng replay buffer lưu dữ liệu từ policy cũ hơn, cho phép tái sử dụng sample, sample-efficient hơn on-policy.

**Vì sao DDPG dễ bị Q overestimation?**
Critic dùng `max_a Q(s', a')` (thông qua actor) trong target → tương tự DQN, `max` gây overestimate. Overestimation trong Q → actor học policy sai.

**TD3 thêm những trick gì để ổn định hơn DDPG?**
1. Clipped double Q-learning: dùng 2 critic, lấy `min` để chống overestimation.
2. Delayed policy update: cập nhật actor ít thường xuyên hơn critic.
3. Target policy smoothing: thêm noise vào action khi tính target để chống overfitting vào Q-value nhọn.

**SAC khác DDPG/TD3 ở điểm nào?**
SAC (Soft Actor-Critic) dùng policy stochastic thay vì deterministic, và thêm entropy regularization vào objective: tối đa hóa `reward + α × entropy(π)`. Khuyến khích khám phá và tránh policy quá deterministic.

**Entropy regularization trong SAC giúp ích gì?**
Giữ policy đủ stochastic → khám phá tốt hơn, tránh bị kẹt ở local optimum, robustness cao hơn với noise môi trường.

**PPO và SAC nên dùng trong hai bối cảnh nào khác nhau?**
PPO: khi muốn đơn giản, ổn định, đặc biệt khi môi trường dễ lấy sample (simulator nhanh). SAC: khi sample efficiency quan trọng, môi trường continuous action, không muốn dùng quá nhiều sample.

**Nếu action là góc lái/throttle liên tục, nên nghĩ tới thuật toán nào trước?**
SAC hoặc TD3 — cả hai xử lý continuous action tốt, SAC thường ổn định và sample-efficient hơn.

**Nếu dữ liệu môi trường đắt, vì sao SAC/TD3 thường hấp dẫn hơn PPO?**
SAC/TD3 là off-policy → dùng replay buffer → tái sử dụng dữ liệu cũ → cần ít sample hơn từ môi trường để học tốt. PPO on-policy phải bỏ data sau mỗi lần cập nhật.

---

## 6. Exploration

**Exploration khác exploitation thế nào?**
Exploration: thử hành động mới chưa biết, có thể phát hiện reward tốt hơn. Exploitation: dùng kiến thức hiện có để chọn action tốt nhất đã biết. Cần cân bằng cả hai.

**Epsilon-greedy hoạt động ra sao?**
Với xác suất `ε`: chọn action ngẫu nhiên (explore). Với xác suất `1-ε`: chọn action tốt nhất theo Q hiện tại (exploit). `ε` thường giảm dần theo thời gian.

**Tại sao exploration khó hơn trong continuous control?**
Không gian action vô hạn → không thể thử hết. Phải thêm noise vào action (ví dụ Gaussian noise trong DDPG, hay dùng entropy trong SAC) thay vì random uniform action.

**Entropy bonus giúp exploration như thế nào?**
Khuyến khích policy có entropy cao (không quá tập trung vào một action), giúp agent thử nhiều action khác nhau trong cùng một state.

**Vì sao agent có thể bị kẹt ở local optimum?**
Nếu policy hội tụ sớm về một chiến lược cho reward trung bình nhưng không khám phá các con đường khác có thể cho reward cao hơn.

**Sparse reward làm exploration khó ra sao?**
Nếu reward chỉ xuất hiện rất hiếm (chỉ khi thắng game), agent có thể không bao giờ nhận reward trong nhiều episode đầu → gradient = 0 → không học được.

**Intrinsic reward / curiosity là gì?**
Thưởng cho agent khi gặp state "mới lạ" chưa thấy nhiều (curiosity-driven). Giúp agent tự động khám phá môi trường ngay cả khi reward ngoại sinh hiếm.

**Đo chất lượng exploration bằng cách nào?**
Xem phần trạm thái (state) đã được thăm trong môi trường, entropy của policy, đa dạng trong trajectory, v.v.

---

## 7. Practical training / debugging

**Vì sao Deep RL khó train hơn supervised learning?**
Không có dataset cố định — dữ liệu thay đổi theo policy. Non-stationary targets. Reward thưa thớt và nhiễu. Feedback loop giữa policy và data. Rất nhạy với hyperparameters.

**Tại sao kết quả RL hay phụ thuộc random seed?**
Quá trình học rất nhạy với khởi tạo ngẫu nhiên, thứ tự dữ liệu, và những ngẫu nhiên nhỏ trong môi trường. Cùng super-theta nhưng seed khác có thể dẫn đến hội tụ rất khác nhau.

**Những hyperparameter nào nhạy nhất trong RL?**
Learning rate, discount factor γ, entropy coefficient (với SAC), network architecture, replay buffer size, batch size, update frequency.

**Vì sao reward scaling lại quan trọng?**
Q-value và gradient phụ thuộc vào scale của reward. Reward quá lớn gây exploding gradient, quá nhỏ làm gradient nhỏ và học chậm. Scale về [−1, 1] hoặc clip reward thường giúp ổn định.

**Reward shaping là gì? Lợi và hại?**
Thêm các phần thưởng phụ để hướng dẫn agent học nhanh hơn. Lợi: tăng tốc học. Hại: nếu shaping sai, agent có thể học khai thác phần thưởng phụ thay vì mục tiêu thật.

**Nếu reward tăng rồi sập, bạn nghi ngờ điều gì?**
Catastrophic forgetting, policy thay đổi quá đột ngột (PPO clipping threshold sai?), replay buffer đã bị "ô nhiễm" bởi data cũ, hoặc learning rate quá lớn.

**Làm sao phân biệt agent "thật sự học" với agent chỉ exploit bug của môi trường?**
Kiểm tra episode được cẩn thận, đánh giá trên nhiều scenario/variant, kiểm tra xem hành vi có hợp lý theo objective thực không, thêm biến thể môi trường để kiểm tra robustness.

**Vì sao evaluation trong RL nên tách khỏi training?**
Để đo hiệu suất thật của policy. Nếu đánh giá trong khi train (với ε-greedy explore), kết quả bị lẫn với hành vi exploration, không phản ánh đúng policy thực.

**Tại sao cần nhiều seeds khi báo cáo kết quả?**
Vì variance giữa seeds rất lớn — một seed có thể cho kết quả rất tốt hoặc rất xấu. Báo cáo trung bình và độ lệch chuẩn trên nhiều seeds mới thực sự phản ánh hiệu quả thuật toán.

**Một training curve đẹp chưa chắc chứng minh được điều gì?**
Curve đẹp trên một seed/environment chưa nói lên được generalization, robustness, hay hiệu quả trên môi trường khác. Có thể là overfitting vào môi trường cụ thể.

---

## 8. Model-based, offline RL, transfer

**Model-free và model-based RL khác nhau thế nào?**
Model-free: học trực tiếp policy/value từ tương tác thực, không biết dynamics môi trường. Model-based: học hoặc dùng một model dynamics `P(s'|s,a)`, dùng model để lập kế hoạch hoặc sinh thêm data giả.

**Khi nào model-based RL đáng cân nhắc hơn model-free?**
Khi sample rất đắt (robot thật, y tế), có thể dùng model để "tưởng tượng" kết quả mà không cần thực tế. Phù hợp khi có thể build được model đủ chính xác.

**Offline RL là gì? Vì sao khó?**
Học policy từ dataset cố định đã thu thập sẵn (không tương tác thêm). Khó vì agent không thể khám phá, và dữ liệu có thể không chứa đầy đủ thông tin cho mọi quyết định.

**Distribution shift trong offline RL là gì?**
Policy học được có thể muốn chọn actions không có trong dataset. Khi đó Q-value cho những actions đó không có dữ liệu để ước lượng chính xác → extrapolation sai.

**Vì sao RL trên dữ liệu cố định dễ bị extrapolation error?**
Q-function được huấn luyện trên distribution của data cũ. Khi policy mới chọn actions nằm ngoài distribution đó, Q(s,a) có thể cho giá trị sai hoàn toàn (thường quá cao), dẫn policy đi sai hướng.

**Transfer learning trong RL là gì?**
Dùng knowledge (policy, value function, hoặc đặc trưng) học được từ task/environment này để tăng tốc học trên task mới liên quan.

**Meta-learning trong RL nhắm tới điều gì?**
Học cách học nhanh: thay vì học giỏi một task, meta-RL học một "prior" giúp agent thích nghi nhanh chóng với task mới chỉ từ vài interaction.

**Vì sao sample efficiency là vấn đề lớn trong robot/thế giới thật?**
Mỗi thử nghiệm với robot tốn thời gian thật, chi phí, và có thể gây hỏng hóc. Không thể train hàng triệu episode như với simulator.

---

## 9. Phân biệt

**V(s) vs Q(s,a) vs A(s,a):** V(s) = kỳ vọng return từ state `s`. Q(s,a) = kỳ vọng return từ `(s,a)`. A(s,a) = Q(s,a) − V(s): action `a` tốt hơn hay kém hơn mức trung bình.

**On-policy vs off-policy:** On: học từ data do policy hiện tại tạo ra. Off: có thể học từ data do policy khác tạo ra (nhờ replay buffer hoặc importance sampling).

**DQN vs policy gradient:** DQN học Q-function, chọn action bằng argmax, off-policy, discrete. Policy gradient tối ưu policy trực tiếp, on-policy, hợp với continuous action.

**DDPG vs TD3:** TD3 thêm 3 trick lên DDPG: clipped double Q, delayed policy update, target policy smoothing → ổn định hơn, ít overestimate hơn.

**TD3 vs SAC:** TD3: policy deterministic, không entropy. SAC: policy stochastic, có entropy regularization → SAC explore tốt hơn và thường ổn định hơn.

**PPO vs SAC:** PPO: on-policy, simple, ổn định, cần nhiều sample. SAC: off-policy, sample-efficient, continuous action, có entropy → khám phá tốt hơn.

**Deterministic vs stochastic policy:** Deterministic: một state → một action cố định. Stochastic: một state → phân phối xác suất trên actions. Stochastic thường hữu ích để explore và với môi trường có đối hand.

**Reward shaping vs objective thật:** Shaping: thêm reward phụ để hướng dẫn. Objective thật: reward môi trường thật sự. Shaping có thể gây agent học target sai nếu không cẩn thận.

**Exploration noise vs environment noise:** Exploration noise: chủ động thêm vào action để khám phá. Environment noise: nhiễu ngẫu nhiên tự nhiên trong môi trường, không thể kiểm soát.

**Sample efficiency vs asymptotic performance:** Sample efficiency: học tốt với ít data. Asymptotic performance: kết quả cuối cùng khi đã train đủ lâu. Hai metrics này thường có đánh đổi.

---

## 10. Câu tự luận

**Vì sao Deep RL cần deep learning, thay vì RL cổ điển tabular?**
State space thực tế (game pixel, robot joint) quá lớn để lưu trong bảng. Neural network cho phép xấp xỉ Q-function hoặc policy trên không gian liên tục và cao chiều, đồng thời tổng quát hóa giữa các state tương tự nhau.

**Pipeline huấn luyện DQN:**
1. Khởi tạo Q-network và target network.
2. Agent quan sát state, chọn action (ε-greedy), nhận reward và state mới.
3. Lưu transition vào replay buffer.
4. Lấy mini-batch ngẫu nhiên từ buffer.
5. Tính Bellman target `r + γ max Q_target(s', a')`.
6. Cập nhật Q-network để minimize TD error.
7. Định kỳ copy Q → target network.
8. Giảm ε theo thời gian.

**Pipeline huấn luyện policy gradient:**
1. Dùng policy hiện tại thu thập trajectories (complete episodes hoặc rollouts).
2. Tính return `G_t` cho từng step.
3. Tính gradient: `∇_θ = Σ_t ∇_θ log π(a_t|s_t) × G_t`.
4. Cập nhật `θ ← θ + α × gradient`.
5. Bỏ data cũ, thu thập lại từ policy mới (on-policy).

**Vì sao actor-critic thường thực dụng hơn vanilla policy gradient?**
Vanilla policy gradient dùng full return → variance cao → cần nhiều episode → chậm học. Actor-critic dùng critic để ước lượng advantage → variance thấp hơn → gradient ổn định hơn → học nhanh hơn với ít sample hơn.

**Tại sao continuous control thường dẫn tới DDPG/TD3/SAC thay vì DQN?**
DQN cần `argmax_a Q(s,a)` — không khả thi khi action là liên tục (vô hạn giá trị). DDPG/TD3 dùng actor network trả về action liên tục trực tiếp. SAC thêm entropy để explore tốt hơn.

**Vì sao PPO được dùng rộng rãi dù không phải lúc nào sample-efficient nhất?**
Đơn giản cài đặt, ổn định, ít hyperparameter nhạy, hoạt động tốt trên nhiều domain khác nhau (game, robot, NLP với RLHF). On-policy nhưng thường đủ nhanh khi simulator rẻ.

**Các nguyên nhân chính khiến Deep RL không ổn định:**
Non-stationary data (distribution thay đổi theo policy), bootstrapping (Q dùng Q để tính target — lỗi tự khuếch đại), correlation giữa consecutive samples, overestimation của Q, và độ nhạy cao với hyperparameters.

**Nếu xây agent lái xe/game/robot, chọn thuật toán nào?**
- Discrete action (game): DQN + Double DQN hoặc PPO.
- Continuous control (xe, robot): SAC hoặc TD3. SAC thường ưu tiên vì stable và sample-efficient.
- Nếu data rất đắt và có data trước: offline RL (IQL, TD3+BC).

---

## 11. Câu vấn đáp ngắn nhưng hay "gài"

**Tại sao `max_a Q(s, a)` gây khó khi action là liên tục?**
Phải giải bài toán tối ưu liên tục `argmax_a Q(s,a)` tại mỗi bước — không có closed-form, phải dùng gradient ascent tốn kém, không thực tiễn cho real-time.

**Tại sao baseline không làm đổi kỳ vọng gradient nhưng lại giảm variance?**
Baseline `b(s)` không phụ thuộc action `a`, nên `E[∇ log π(a|s) × b(s)] = b(s) × E[∇ log π(a|s)] = 0` — không thay đổi kỳ vọng gradient. Nhưng `b(s)` tương quan với return → trừ đi làm signal thực sự (phần variance thật của return) nhỏ hơn.

**Tại sao replay buffer hợp với off-policy hơn on-policy?**
On-policy yêu cầu data đến từ chính sách hiện tại. Replay buffer chứa data cũ từ nhiều policy khác nhau — vi phạm assumption của on-policy. Off-policy không yêu cầu điều này nên tái sử dụng data từ buffer hợp lệ.

**Vì sao entropy lớn có thể tốt lúc đầu nhưng không nên quá lớn mãi?**
Entropy lớn ban đầu giúp explore rộng. Nhưng khi đã hiểu môi trường, policy nên trở nên confident hơn (entropy giảm) để khai thác kiến thức tốt. SAC dùng α có thể điều chỉnh để tự cân bằng.

**Tại sao reward function kém có thể khiến agent học hành vi sai dù thuật toán đúng?**
RL tối ưu đúng reward function được cho. Nếu reward function không phản ánh đúng mục tiêu thật, agent sẽ học tối ưu proxy đó thay vì mục tiêu người thiết kế muốn — được gọi là "reward hacking".
