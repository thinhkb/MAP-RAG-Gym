# MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation

Yiqun Chen1*, Erhan Zhang1*, Lingyong Yan2, Shuaiqiang Wang2 Jizhou Huang2, Dawei Yin2, Jiaxin Mao1† 

1Renmin University of China, 2Baidu Inc. 

chenyiqun990321@ruc.edu.cn, erhanzhang@ruc.edu.cn, maojiaxin@gmail.com 

## Abstract

In question-answering (QA) systems, Retrieval-Augmented Generation (RAG) has become pivotal in enhancing response accuracy and reducing hallucination issues. The architecture of RAG systems varies significantly, encompassing singleround RAG, iterative RAG, and reasoning RAG, each tailored to address different types of queries. Due to the varying complexity of real-world queries, a fixed RAG pipeline often struggles to balance performance and cost efficiency across different queries. To address this challenge, we propose an adaptive RAG framework called MAO-ARAG, which leverages multi-agent orchestration. Our adaptive RAG is conceived as a multi-turn framework. Specifically, we define multiple executor agents, representing typical RAG modules such as query reformulation agents, document selection agent, and generation agents. A planner agent intelligently selects and integrates the appropriate agents from these executors into a suitable workflow tailored for each query, striving for high-quality answers while maintaining reasonable costs. During each turn, the planner agent is trained using reinforcement learning, guided by an outcome-based reward (F1 score) and a cost-based penalty, continuously improving answer quality while keeping costs within a reasonable range. Experiments conducted on multiple QA datasets demonstrate that our approach, which dynamically plans workflows for each query, not only achieves high answer quality but also maintains both cost and latency within acceptable limits.The code of MAO-ARAG is on https://github.com/chenyiqun/Agentic-RAG. 

## 1 Introduction

Large Language Models (LLMs) have been extensively used for various tasks, including question answering (Asai et al. 2023; Khattab et al. 2022), information retrieval (Sun et al. 2023; Zhang et al. 2024; Chen et al. 2024), different types of reasoning (Huang and Chang 2022; Hao et al. 2023), and evaluation (Gong and Mao 2023; Fu et al. 2023). Despite their wide applicability, LLMs face limitations due to their inability to update internal knowledge promptly after pre-training, making them susceptible to producing outdated or inaccurate information (Zhao et al. 2023). To address these limitations, Retrieval-Augmented Generation (RAG) systems have been developed to boost the generative performance of LLMs by integrating relevant information from external knowledge sources, with an inherently modular architecture (Gao et al. 2024b) that allows for customization to specific tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/bb7c995ee201acbf0e32e20f794b70e73341646b4545ca52541c0985b0200bfb.jpg)



Figure 1: The appropriate workflows for different types of queries are highly heterogeneous.


A naive RAG pipeline typically includes a retrieval model (Wang et al. 2022; Xiao et al. 2024) that retrieves candidate texts, and a LLM that generates answers based on the retrieved texts. Beyond this fundamental structure, other advanced RAG pipelines incorporate additional modules such as query rewriting (Ma et al. 2023; Chen et al. 2025), document selection (Ke et al. 2024; Li et al. 2024a), and selfreflection (Asai et al. 2023). Recently, methods like Searcho1 (Li et al. 2025) and Search-r1 (Jin et al. 2025) have further advanced the capabilities of RAG systems by integrating reasoning processes. 

These RAG systems, with their diverse components, are suited to different scenarios. Naive RAG are ideal for straightforward queries, offering benefits such as lower costs and reduced latency. However, they tend to struggle with more complex queries. On the other hand, more sophisticated RAG systems excel in handling intricate questions but at the cost of increased LLM usage and higher latency. In real-world QA systems, queries often vary widely in type and difficulty. And as shown in Figure 1, the ideal workflows vary across different types of queries. Therefore, it is challenging for a fixed RAG pipeline to consistently deliver high-quality answers across diverse queries while keeping costs—such as LLM token usage and system latency—within a reasonable range. 

To address this challenge, we propose an Adaptive multiturn RAG framework called MAO-ARAG, utilizing Multi-Agent Orchestration. Within this framework, we define multiple executor agents comprising common modules found in existing RAG systems, such as query reformulation, retriever, document selection, and answer generation, etc. At the core of this framework lies a planner agent that selects suitable executors for each query and orchestrates them to form a query-specific workflow. To improve the effectiveness and efficiency of the planner agent’s orchestration, we adopt the Proximal Policy Optimization (PPO) algorithm (Schulman et al. 2017), guided by an outcome-based reward (F1 score) and a cost-based penalty. This approach ensures that the constructed pipeline achieves high answer quality while keeping costs, such as resource consumption and latency, within reasonable bounds. 

Our main contributions are as follows: 

• We propose MAO-ARAG, a novel multi-agent framework for adaptive RAG that features a planner agent to dynamically select and compose multiple executor agents—modular components commonly used in RAG systems—into a query-specific workflow. 

• We propose a PPO-based training algorithm that incorporates outcome-based rewards and cost-based penalties to improve the planner agent’s ability to balance answer quality and computational cost. 

• We conduct extensive experiments on multiple QA benchmarks to validate that the proposed MAO-ARAG framework can tailor a suitable RAG pipeline for each query, achieving high answer quality and maintaining appropriate cost. 

## 2 Related Work

## 2.1 Different Modules in RAG System

Retrieval Model plays a fundamental role in the RAG system, which supplies external knowledge to the LLM-based generator to generate final answers. Within the context of retrieval models for RAG, BM25 (Robertson and Walker 1994) stands out as a traditional yet effective sparse retrieval model. In contrast, Contriever (Izacard et al. 2021), BGE (Wang et al. 2022), and E5 (Xiao et al. 2024) are designed to produce dense embeddings, making them effective dense retrieval models. Lastly, ColBERT (Khattab and Zaharia 2020) improves information retrieval accuracy and efficiency by employing multi-vector representations and a “late interaction” mechanism. 

Query Reformulation is to rewrite or decompose initial query in RAG, and are introduced in RRR (Ma et al. 2023) and DMQR-RAG (Li et al. 2024b). 

Document Selection is to select helpful information from the noise candidate documents. BGM (Ke et al. 2024) and RAG-DDR (Li et al. 2024a) both utilize this module. 

Answer Generator is responsible to output the answer to the input query. There are many open-source LLMs, such as Deepseek (Bi et al. 2024), Llama (Grattafiori et al. 2024), Qwen (Yang et al. 2025), and many close-source LLMs, such as GPT (Brown et al. 2020) and Gemini (Team et al. 2023), which can be considered as an answer generator. 

## 2.2 Typical Workflows in RAG System

Single-Round RAG The modules in single-turn RAG are organized in a linear way. RRR (Ma et al. 2023) propose a Rewrite-Retrieve-Read framework and BGM (Ke et al. 2024) introduce a selection-generation paradigm. In addition, RAG-DDR (Li et al. 2024a) and MMOA-RAG (Chen et al. 2025) also propose a linear RAG pipeline. All these method utilized reinforcement learning algorithm to optimize single or multiple modules in RAG pipelines. 

Iterative RAG The RAG pipeline in iterative RAG is a loop structure. ITER-RETGEN (Shao et al. 2023) is a method that improves retrieval-augmented large language models by iteratively integrating retrieval and generation processes. SELF-RAG (Asai et al. 2023) boosts the quality and factual accuracy of language models through a process of self-reflective retrieval and generation. DRAGIN (Su et al. 2024) is a framework that dynamically addresses the real-time information needs of large language models during text generation, enhancing their performance on tasks that require extensive knowledge. SMARTRAG (Gao et al. 2024a) utilizes PPO to optimize an iterative RAG framework with answer-based reward. 

Reasoning RAG Search-o1 (Li et al. 2025) enhances the RAG utilizing reasoning ability of LLM. After Deepseek-r1 (Guo et al. 2025), some works introduce training the reasoning LLM to improve the performance in RAG. For example, Search-r1 (Jin et al. 2025) and R1-Searcher (Song et al. 2025) both use answer-based reward to improve the reasoning in RAG. 

## 3 Methods

## 3.1 Modeling RAG as a Multiagent Semi-Markov Decision Process

To capture the requirements of orchestrating heterogeneous agents across varying scenarios, we model the RAG system as a Multiagent Semi-Markov Decision Process (MSMDP) (Ghavamzadeh, Mahadevan, and Makar 2006), which effectively captures coordination among agents with distinct roles. 

An MSMDP extends the traditional Markov Decision Process (MDP) (Sutton, Barto et al. 1998) by accommodating multiple agents and allowing for actions of variable durations. Formally, an MSMDP can be defined as a tuple $\langle S , A , P , R , \gamma , \dot { T } \rangle$ . S is the state space. $A \quad =$ $\left\{ A _ { 1 } , A _ { 2 } , \ldots , A _ { n } \right\}$ is a set of action sets, where $A _ { i }$ is the set of actions available to agent i. $P : S \times A \times S  [ 0 , 1 ]$ is the state transition probability function. $R : S \times A { \stackrel { \cdot } { \to } } \mathbb { R }$ is the reward function, providing feedback to the agents based on the current state and actions taken. $T : S \times A \to \mathbb { R } ^ { + }$ is a function representing the duration of executing an action. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/0217c7beaf02d72c2a2b961686645e46ec1de37efa544adb9aabc47560a73360.jpg)



Figure 2: The overall framework of MAO-ARAG.


To implement an adaptive RAG, we introduce MAO-ARAG, which employs a planner to coordinate multiple executors1, as illustrated in Figure 2. 

The Planner is responsible for designing an appropriate workflow for a given question or a rewritten sub-question, and the workflow is composed of a subset of the executors. 

The executors encompass several commonly used modules in the current modular RAG process, mainly including: 

• Query Decomposition Serial (QDS): This module serially decomposes a given question into several subquestions that have sequential dependencies. The answer to a later sub-question often depends on the answers to preceding ones. 

• Query Decomposition Parallel (QDP): This module decomposes a given question into multiple independent sub-questions that can be searched in parallel. 

• Query Rewriter (QR): This module rewrites a question into a clearer and more searchable version. 

• Document Selector (DS): Given a question and multiple candidate documents, this module selects documents that are helpful for answering the question and excludes those that are not. 

• Retrieval Agent (RA): This is a search engine that takes a question as input and returns the top k most relevant candidate documents from a corpus. 

• Answer Generator (AG): This module generates an answer to a given question, which may be informed by reference documents or generated independently. 

• Answer Summarization (AS): Based on the subquestions and their respective sub-answers, this module provides an answer to the initial question. 

In MSMDP, since the duration of an action T is not fixed, the MSMDP allows for effective coordination and optimization across different time scales and agent roles, making it a suitable way to model an adaptive RAG. By leveraging the MSMDP framework, our MAO-ARAG efficiently integrates the decision-making processes of the planner agent and executor agents. 

## 3.2 Essential Elements of RL

In our framework, the workflow plays a crucial role in determining the final answers, and since the planner is responsible for generating this workflow, optimizing the planner becomes essential and important. The framework involves multiple rounds during the whole rollout process, with each round requiring the planner to design a appropriate workflow for the given (sub-)question. Moreover, our optimization goals are not limited to enhancing the answer qualities; they also include reducing cost and latency, making this a multi-objective optimization problem. Taking these factors into account, we employ a reinforcement learning approach (PPO algorithm) to optimize the parameters of planner agent with an outcome-based reward and cost-based penalty terms. 

In the following, we will introduce the essential elements of planner, which mainly contains Observation, Action Space and Reward Function: 

• Observation of planner is defined as Equation (1), which contains the prompt of planner $P r o m p t _ { p l a n n e r }$ and a given question q. And q is a initial question or a subquestion. 

$$
O _ {\text { planner }} = \left\{\text { Prompt } _ {\text { planner }}, q \right\} \tag {1}
$$

• Action Space of planner is the abbreviation of each executors, which is defined as Equation (2). The output of planner is a combination of the abbreviations of executors in the action space. 

$$
A _ {\text { planner }} = \{\text { QDS }, \text { QDP }, \text { QR }, \text { DS }, \text { RA }, \text { AG }, \text { AS } \} \tag {2}
$$

• Reward Function of the planner comprises three components. The first component, shown in Equation (3), is the F1 score calculated between the predicted answer, apredicted, generated by the RAG, and the golden answer, $a _ { \mathrm { g o l d e n } } .$ This F1 score serves as one of the key performance metrics we strive to optimize and is also the Final Reward illustrated in Figure 2. 

$$
R _ {f 1} = F 1 (a _ {\text { predicted }}, a _ {\text { golden }}) \tag {3}
$$

The second component is a penalty term concerning the cost, denoted as Cost Penalty (CP) in Figure 2, which is defined as Equation (4). 

$$
R _ {C P} = \text { Token } _ {\text { cost }} + \text { Turn } _ {\text { cost }} + \mathbb {I} (S) \tag {4}
$$

In Equation (4), $T o k e n _ { \mathrm { c o s t } }$ represents the token cost of the workflow provided by the planner, scaled to a range between 0 and 1. Similarly, $T u r n _ { \mathrm { c o s t } }$ denotes the cost associated with latency. If a given workflow brings more turns later, the $T u r n _ { \mathrm { c o s t } }$ will be larger. But $T u r n _ { \mathrm { c o s t } }$ is also scaled between 0 to 1. The function I(S) is an indicator for the search engine call. If the Retrieval Agent (RA) is utilized in the workflow, then $\mathbb { I } ( S ) = 1 ;$ ; otherwise, $\mathbb { I } ( S ) = 0 .$ . As for the specifics of how and why these three cost penalties are scaled, you can refer to Appendix A. 

The third component concerns the penalty term related to the workflow format, denoted as Format Penalty (FP) in Figure 2, which is defined as Equation (5). 

$$
R _ {F P} = \mathbb {I} (\text { workflow }) \tag {5}
$$

In Equation (5), only when the workflow format is correct and executable is $R _ { F P }$ equal to 0; otherwise, it is 1. 

Finally, the total reward of the planner can be defined as Equation (6), which consists of $R _ { f 1 } , R _ { C P } ,$ , and $R _ { F P } .$ . α is a hyperparameter. 

$$
R _ {p l a n n e r} = R _ {f 1} - \alpha \cdot R _ {C P} - R _ {F P} \tag {6}
$$

## 3.3 Traning Process of RL

As illustrated in Figure 2, we model the entire rollout as a multi-turn process and utilize PPO algorithm to optimize the planner to get better evaluation metrics. The training process of MAO-ARAG is shown in Algorithm 1. The parameters for the actor model and the critic model are denoted as θ and ϕ, respectively, and the reference model is denoted as $\theta _ { i n i t }$ In each turn, a planner is responsible for designing an adaptive workflow w based on either the initial problem $q _ { i n i t }$ or its reformulated sub-question $q _ { s u b }$ . Subsequently, the executors implement this workflow. Upon the completion of all turns, a predicted answer $a _ { \mathrm { p r e d i e c t e d } }$ is obtained, which is then evaluated against the golden answer $a _ { \mathrm { g o l d e n } }$ using the F1 score. This F1 score serves as a shared reward across all turns. Additionally, each turn incorporates a Format Penalty (FP) and a Cost Penalty (CP) to make the workflow executable and balance the overall cost. Following this, we employ the PPO algorithm to update the planner’s parameters based on the data collected in each turn. The overall loss function of PPO, ${ \mathcal { L } } ( \theta , \phi )$ , consists of two terms: ${ \mathcal { L } } _ { \mathrm { A c t o r } } ( \theta )$ and ${ \mathcal { L } } _ { \mathrm { C r i t i c } } ( \phi )$ : 


Algorithm 1: The Training Process of MAO-ARAG


Initialize: The parameters of the Actor model $\theta$ , the Critic model $\phi$ , the initial model $\theta_{\mathrm{init}}$ , and a replay buffer $\mathcal{M} = \emptyset$ .

Inputs: Dataset with initial questions $q$ and corresponding golden answers $Ans_{golden}$ for batch $\leftarrow 1$ to N_batch do

// Collect Data

for each question $q_{init} \in batch$ do

// Rollout Multiple Turns

for turn $_i$ in MAX_TURN do

// Planner

Determine the given question $q$ to planner. ( $q$ is $q_{init}$ or its sub-question $q_{sub}$ .)

Construct observation $O_{planner}^i$ according to Equation (1).

Get the workflow $w$ to the given question.

Get the Format Penalty (FP) $R_{FP}^i$ for this turn $_i$ .

// Execute the workflow

Following the workflow $w$ , execute each executor involved.

Get the Cost Penalty (CP) $R_{CP}^i$ for this turn $_i$ .

Store tuple $\mathcal{T}_i = (O_{planner}^i, w_i, R_{FP}^i, R_{CP}^i)$ in the replay buffer $\mathcal{M}$ Get the predicted answer $a_{predicted}$ for $q_{init}$ .

Compute the F1 score as $R_{f1}$ for $q_{init}$ .

// Update the data

for each turn $_i$ in MAX_TURN do

Calculate the total reward $R_{planner}^i$ for turn $_i$ according to Equation (6).

Update the tuple $\mathcal{T}_i = (O_{planner}^i, w_i, R_{planner}^i)$ in the replay buffer $\mathcal{M}$ .

// Policy and Value Optimization

for each question $q \in batch$ do

Compute the advantage function $\hat{A}_{\pi_\theta}^t$ using GAE

Calculate the loss of the Actor $\mathcal{L}_{\text{Actor}}(\theta)$ and Critic model $\mathcal{L}_{\text{Critic}}(\phi)$ Update the parameters of models through the overall loss function $\mathcal{L}(\theta, \phi)$ in Equation (7)

Clear the replay buffer $\mathcal{M}$ to $\emptyset$ Output: A well-trained planner: Actor model with parameters $\theta_{\text{trained}}$ 

$$
\mathcal {L} (\theta , \phi) = \mathcal {L} _ {\text { Actor }} (\theta) + \mathcal {L} _ {\text { Critic }} (\phi) \tag {7}
$$

The actor loss ${ \mathcal { L } } _ { \mathrm { A c t o r } } ( \theta )$ can be defined as Equation (8). The term rt = $\begin{array} { r } { r _ { t } = \frac { \pi _ { \theta } \left( a _ { t } \vert s _ { t } \right) } { \pi _ { \theta _ { o l d } } \left( a _ { t } \vert s _ { t } \right) } } \end{array}$ denotes the importance sampling ratio, which measures the difference between the new and old policies. The expression $\begin{array} { r } { \hat { A } _ { \pi _ { \theta } } ^ { t } = \sum _ { l = 0 } ^ { \infty } ( \gamma \lambda ) ^ { l } \delta _ { t + l } } \end{array}$ is the advantage function, estimated using GAE (Schulman et al. 2015). The variable $\delta _ { t } = R ( s _ { t } , a _ { t } ) \bar { + } \gamma V _ { \phi } ( s _ { t + 1 } ) - V _ { \phi } ( s _ { t } )$ is known as the temporal difference (TD) error at time step t. 

$$
\mathcal {L} _ {\text { Actor }} (\theta) = \sum_ {t} \min \left(r _ {t} \hat {A} _ {\pi_ {\theta}} ^ {t}, \operatorname{clip} \left(r _ {t}, 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {\pi_ {\theta}} ^ {t}\right) \tag {8}
$$

The Equation (9) is similar with the reward in PPO training for LLM (Ouyang et al. 2022). $R _ { p l a n n e r }$ contains three components defined in Equation (6). 

$$
R (s _ {t}, a _ {t}) = \left\{ \begin{array}{l l} 0, & \text { if } t <   T \\ R _ {\text { planner }} - \beta \cdot \log \left(\frac {\pi_ {\theta} (w | O _ {\text { planner }})}{\pi_ {\theta_ {\text { init }}} (w | O _ {\text { planner }})}\right), & \text { if } t = T \end{array} \right. \tag {9}
$$

The critic loss ${ \mathcal { L } } _ { \mathrm { C r i t i c } } ( \phi )$ is defined in Equation (10), employing a clipping operation similar to the actor loss. Here, $\Delta \dot { V _ { t } } = V _ { \phi } ^ { t } - V _ { \mathrm { t a r g e t } } ^ {  }$ $V _ { \phi } ^ { t } = V _ { \phi } ( s _ { t } )$ $V _ { \mathrm { t a r g e t } } ^ { t }$ represents the cumulative return and $s _ { t }$ is the state-values. 

$$
\mathcal {L} _ {\text { Critic }} (\phi) = \sum_ {t} \max \left[ \left(\Delta V _ {t}\right) ^ {2}, \left(\operatorname{clip} \left(V _ {\phi} ^ {t}, V _ {\phi_ {\text { old }}} ^ {t} \pm \epsilon\right) - V _ {\text { target }} ^ {t}\right) ^ {2} \right] \tag {10}
$$

After multiple steps of training, we can obtain a welltrained planner agent that can customize an appropriate workflow for each query. 

## 4 Experiments

Our experiments mainly focus on the following research questions: 

• RQ.1: Can MAO-ARAG outperform the existing common RAG pipeline? 

• RQ.2: Is MAO-ARAG an efficient method? In other words, can MAO-ARAG achieve good performance while keeping costs within a reasonable range? 

• RQ.3: How does α in Equation (6) affect the learned strategies? 

• RQ.4: Can we use different LLMs as the planner and executors in MAO-ARAG? For example, can we use a smaller planner for efficiency? Can we leverage alternative LLMs to support the executors for different tradeoffs in effectiveness and cost? 

## 4.1 Experimental Setup

Datasets To evaluate the effectiveness of our MAO-ARAG, we conduct experiments on a diverse set of opendomain question answering (QA) benchmarks: 

• Single-hop QA: We include Natural Questions (NQ) (Kwiatkowski et al. 2019), PopQA (Mallen et al. 2022), and AmbigQA (Min et al. 2020). 

• Multi-hop QA: We also use HotpotQA (Yang et al. 2018), 2WikiMultiHopQA (Ho et al. 2020), Musique (Trivedi et al. 2022), and Bamboogle (Press et al. 2022) to test the ability of different methods. 

Corpus and Retriever For all retrieval-based methods, we utilize Wikipedia as the corpus (Karpukhin et al. 2020). Retriever is performed using E5 (Wang et al. 2022). 

Evaluation Metrics We evaluate model performance using F1 score. We also utilize the token cost, retriever call times, and turn number as the cost metrics. 

Models we mainly employ Qwen2.5-7B-Instruct (Team 2024) as the planner, responsible for analyzing the input query and generating an appropriate workflow. For the executor agents, we utilize GPT-4o-Mini (Hurst et al. 2024) as the backbone to perform the corresponding functions. 

Baselines We compare our approach with different types of baselines2: 

• Singel-Round RAG: (1) LLM w/o RAG: Answers are generated solely based on the internal knowledge of the LLM. (2) Vanilla RAG: A conventional RAG setup where retrieved documents are used to generate answers. (3) RRR (Ma et al. 2023): Introduce query reformulation in RAG. (4) BGM (Ke et al. 2024): Add a documents selection module in RAG pipeline. (5) MMOA-RAG (Chen et al. 2025): The workflow contains query rewriter, retriever, document selector, answer generator. 

• Iterative RAG: (6) Self-RAG (Asai et al.): Combines adaptive retrieval with self-reflection to enhance answer reliability and precision. 

• Agentic RAG: (7) Search-o1 (Li et al. 2025): Incorporate an agentic retrieval mechanism and a dynamic workflow. (8) Search-r1 (Jin et al. 2025): Utilize RL reasoning training to enhance the agentic RAG. 

## 4.2 Performance of Different Methods (RQ.1)

We evaluated various algorithms across the seven datasets presented in Table 1, focusing on their F1 scores for comparison. Our training utilized only 2400 question-answer pairs from the NQ training dataset and 4800 pairs from the HotpotQA training dataset, followed by testing on all seven datasets. To reduce testing costs, we randomly selected 1000 question-answer pairs from the official test sets of each dataset (with Bamboogle having only 125 pairs). 

In Table 1, “MAO-ARAG w/o train” signifies the use of the untrained Qwen2.5-7B-Instruct as the planner. Meanwhile, “MAO-RAG” represents our method, trained via RL training, where the α hyperparameter in Equation 6 is set to zero, indicating an exclusive focus on optimizing the F1 score without considering cost reduction. 

Table 1 reveals that the average F1 score of MAO-ARAG w/o train across the seven datasets ranks just below Searcho1 among the baselines, suggesting that even an untrained planner can effectively organize and manage executors. Moreover, our MAO-ARAG method achieved the highest performance on 5 out of the 7 datasets, with an average F1 score of 52.91. This is 3.08 points higher than the best baseline, Search-o1, which scored 49.83, and 8.53 points higher than MAO-ARAG w/o train, which had a score of 44.38. These results highlight the effectiveness of our optimization mechanism for the planner in multi-turn adaptive RAG, demonstrating its capability to effectively select and arrange executors to achieve the goal of optimizing the F1 score.3 


Table 1: F1 performance (%) of various methods across datasets. The font with the highest score in each dataset is bold, and the second highest score is underlined. ∆ indicates the improvement of MAO-ARAG over the best baseline.


<table><tr><td>Methods</td><td>NQ</td><td>PopQA</td><td>AmbigQA</td><td>HotpotQA</td><td>2Wiki</td><td>Musique</td><td>Bamboogle</td><td>Average</td></tr><tr><td>LLM w/o RAG</td><td>39.96</td><td>30.99</td><td>49.90</td><td>42.38</td><td>33.49</td><td>20.74</td><td>36.15</td><td>36.23</td></tr><tr><td>Vanilla RAG</td><td>48.02</td><td>44.23</td><td>59.04</td><td>49.54</td><td>37.62</td><td>25.66</td><td>43.45</td><td>43.94</td></tr><tr><td>RRR (Ma et al. 2023)</td><td>46.27</td><td>41.59</td><td>56.15</td><td>43.14</td><td>29.77</td><td>23.21</td><td>37.81</td><td>39.71</td></tr><tr><td>BGM (Ke et al. 2024)</td><td>48.41</td><td>45.39</td><td>59.25</td><td>49.58</td><td>36.79</td><td>25.60</td><td>44.10</td><td>44.16</td></tr><tr><td>MMOA-RAG (Chen et al. 2025)</td><td>46.88</td><td>40.26</td><td>55.88</td><td>43.19</td><td>30.40</td><td>21.78</td><td>36.53</td><td>39.28</td></tr><tr><td>Self-RAG (Asai et al. 2023)</td><td>41.60</td><td>34.25</td><td>52.06</td><td>47.94</td><td>39.53</td><td>32.88</td><td>56.33</td><td>43.51</td></tr><tr><td>Search-r1 (Jin et al. 2025)</td><td>42.22</td><td>43.35</td><td>52.50</td><td>44.44</td><td>34.13</td><td>21.44</td><td>37.83</td><td>39.42</td></tr><tr><td>Search-o1 (Li et al. 2025)</td><td>46.68</td><td>43.12</td><td>56.93</td><td>53.75</td><td>47.26</td><td>39.51</td><td>61.58</td><td>49.83</td></tr><tr><td>MAO-ARAG w/o train</td><td>50.57</td><td>32.73</td><td>55.15</td><td>49.68</td><td>40.75</td><td>32.36</td><td>49.41</td><td>44.38</td></tr><tr><td>MAO-ARAG</td><td>54.50</td><td>54.16</td><td>57.80</td><td>53.80</td><td>47.69</td><td>37.33</td><td>65.09</td><td>52.91</td></tr><tr><td>Δ</td><td>+6.09</td><td>+8.77</td><td>-1.45</td><td>+0.05</td><td>+0.43</td><td>-2.18</td><td>+3.51</td><td>+3.08</td></tr></table>

## 4.3 Cost-Performance Trade-Off (RQ.2)

The F1 score is used to evaluate the quality of predicted answers apredicted across different methods, but the cost of generating these predictions is also important. In this study, we assessed three metrics related to prediction cost: 

• Token Cost: Represents the average cost of tokens consumed to answer predictions (in USD per query). 

• Retrieval Calls: Indicates the average number of retrieval calls made (calls per query). 

• Turns: The average number of turns required to complete a query (turns per query). 

Figure 3 illustrates the relationship between the performance metric (F1 score) and the three prediction cost metrics.4 A higher value on the horizontal axis signifies greater cost consumption, while a higher value on the vertical axis indicates better performance. Therefore, methods positioned closer to the top-left corner of the graphs achieve superior results with fewer resource expenditures. 

Our MAO-ARAG α = 0 achieved the highest F1 score, yet its cost metrics were not the highest. In Figure 3, MAO-ARAG with different α forms a black dotted line, which is relatively close to the top-left corner, indicating that MAO-ARAG can achieve optimal performance at a relatively reasonable cost. Notably, although MAO-ARAG w/o train and Search-o1 have similar F1 scores, the cost metrics for MAO-ARAG w/o train are significantly lower than those for Search-o1. This suggests that our proposed architecture, which separates workflow planning and execution, inherently promotes more efficient resource use. While the costs for MAO-ARAG increase moderately after RL training compared to MAO-ARAG w/o train, its performance sees a substantial improvement of approximately 4%. 

The performance of Search-r1 is somewhat inferior compared to our method. This discrepancy arises from the fact that in Search-r1, the implicit workflow planning and execution are tightly coupled, with all processes executed by a trainable agent based on an open-source LLM. The necessity for the model to be trainable, combined with the integration of planning and execution, results in suboptimal performance for Search-r1. This also highlights the advantages of our framework, which distinctly separates planning and execution while enabling the planner agent to be trainable. 

## 4.4 Effect of Different Cost Weight α (RQ.3)

In reinforcement learning, the ultimate strategy adopted by an agent is highly correlated with the reward function. Within our MAO-ARAG framework, the reward function is defined in Equation (6), where the hyperparameter α governs the cost-based penalty term. By tuning the value of α, we can achieve a balance between the effectiveness and the cost of the RAG pipeline. Theoretically, as α increases, the penalty on cost intensifies, which may degrade the planner’s workflow performance while reducing the cost associated with obtaining answers (such as token cost, retriever call times, and latency). Conversely, a decrease in α enhances performance but incurs higher costs. 

Figure 4 illustrates the line graphs depicting the performance metric F1 score alongside three cost indicators under varying α values. It is evident that as α increases, there is a general decline in overall performance (F1 score), accompanied by a reduction in the three cost metrics due to the heightened penalty. Interestingly, when α exceeds 0.2, a rapid decline in various performance metrics occur. This phenomenon can be attributed to the fact that, as α increases, the cost-based penalty term in the reward function becomes disproportionately large. Consequently, the trained planner tends to generate overly simplistic workflows. 

Additionally, it can be observed that the curves in Figure 4 exhibit fluctuations. This may be due to the following reasons: (1) The limited number of executors defined may mean that the optimal workflow is composed of fewer executors, introducing significant uncertainty and causing fluctuations. (2) To simplify the definition of the Cost Penalty term $R _ { C P }$ in Equation (4), we scaled the token cost $T o k e n _ { c o s t } ,$ , turn numbers cost $T u r n _ { c o s t }$ , and search engine call cost I(S) to a range of [0, 1]. However, there might be inherent weights among different cost penalty terms that ought to be considered. This coarse definition of the Cost Penalty term $R _ { C P }$ could also contribute to the observed fluctuations, indicating a potential area for future refinement. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/bef26cfb5805d873822aed0625554c2642d733ecbd4096d22ccc367129037d6d.jpg)



(a) F1 score vs. Token Cost


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/22db06d55d5f1e6b161b13f41e457b1fa05f6cb160d44d209340dff4a8d96f04.jpg)



(b) F1 score vs. Retrieval Calls


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/eca0ee4f4444030d2feade19313f4419bda048c8d2f7c1a69c0eb0f9a4a57f41.jpg)



(c) F1 score vs. Turns



Figure 3: F1 score vs. cost metrics of different methods. All data in this figure is the average of NQ and HotpotQA datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/abf15d341af76a982ed494b48e7a9ac9fbd40040671fd193785ec0df1622128f.jpg)



(a) F1 Score vs. α


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/6ee3a86230d5b7fcd73c26bbd0642f8519c6f8e42799574aa18253bc4d07fe0e.jpg)



(b) Token Cost vs. α


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/79f6440206e260dba446bef4e543225695609db0a8ccf615674fbbb83914ce4a.jpg)



(c) Retrieval Calls vs. α


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-10/3e82ec74-8023-4bf6-a2b7-8454d29d998e/eee4c8cd93482e66db1b9b7662861cac8835438cef9460129e3a406c37613ffe.jpg)



(d) Turns vs. α



Figure 4: The metrics vs. α of different methods. All data in this figure is the average of NQ and HopotQA datasets.


## 4.5 Smaller Planner and Different Executor Backbone (RQ.4)

In this section, we explore the feasibility of using smaller models as planners and conduct preliminary experiments using alternative APIs as executor backbones. 

We initiate our study by distilling the trained 7B planner model into 1.5B and 0.5B models using supervised finetuning (+SFT), followed by PPO training (+PPO). As shown in Table 2, the performance and cost metrics of these smaller planners (+SFT and +PPO) closely match those of the 7B model (+PPO). This indicates that larger planners can be effectively distilled into smaller models capable of performing the planner’s role just as well. 

Furthermore, while previous experiments predominantly used GPT-4o-mini for the executor agents, we replaced this backbone with GPT-3.5-turbo and GPT-4.1-nano, respectively. From Table 2, we can see that both alternatives achieve similar F1 scores, albeit lower than the 7B (+PPO) 


Table 2: Smaller planner and other different backbones of executors. (Average metrics of NQ and HotpotQA datasets)


<table><tr><td>Model &amp; Backbone</td><td>F1 Score</td><td>Token Cost</td><td>Retrieve Times</td><td>Turn Number</td></tr><tr><td>7B w/o train</td><td>50.13</td><td>0.00064</td><td>1.56</td><td>2.02</td></tr><tr><td>7B (+PPO)</td><td>54.15</td><td>0.00112</td><td>2.27</td><td>2.77</td></tr><tr><td colspan="5">Smaller Planner</td></tr><tr><td>1.5B (+SFT)</td><td>53.81</td><td>0.00102</td><td>2.13</td><td>2.59</td></tr><tr><td>1.5B (+PPO)</td><td>53.91</td><td>0.00095</td><td>2.06</td><td>2.48</td></tr><tr><td>0.5B (+SFT)</td><td>53.64</td><td>0.00101</td><td>2.10</td><td>2.54</td></tr><tr><td>0.5B (+PPO)</td><td>53.92</td><td>0.00111</td><td>2.26</td><td>2.76</td></tr><tr><td colspan="5">Different Executor Backbone</td></tr><tr><td>GPT-3.5-turbo</td><td>48.08</td><td>0.00328</td><td>1.63</td><td>2.24</td></tr><tr><td>GPT-4.1-nano</td><td>47.43</td><td>0.00051</td><td>1.68</td><td>2.13</td></tr></table>

using GPT-4o-mini as the executors’ backbone. However, due to the lower cost of the GPT-4.1-nano API, its token cost is only 0.00051, less than half of the 0.00112 incurred by the 7B (+PPO). Conversely, GPT-3.5-turbo, being an outdated model, not only results in a lower F1 score but also incurs a higher token cost. 

The experiments in this section demonstrate that we can use smaller planners and more cost-effective APIs as executors’ backbones, achieving a more favorable balance between effectiveness and cost. 

## 5 Conclusion

In this paper, we proposed MAO-ARAG, a novel multiagent orchestration framework for adaptive RAG in QA systems. MAO-ARAG dynamically constructs appropriate workflows for diverse queries, leveraging multiple executor agents, including query reformulation, document selection, and answer generation modules. These agents are orchestrated by a planner agent optimized using RL with a reward function that balances answer quality and cost metrics. Through extensive experiments on a variety of single-hop and multi-hop QA datasets, we demonstrated that MAO-ARAG outperforms existing RAG pipelines, achieving a better balance between effectiveness and cost. 

Future work may focus on refining cost penalties to better balance performance and cost. We also plan to use multiple optional APIs simultaneously as executor backbones, aiming for better results at a lower cost. 

## References



Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H. 2023. Self-rag: Learning to retrieve, generate, and critique through self-reflection. arXiv preprint arXiv:2310.11511. 





Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H. S.-R. ???? Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv 2023. arXiv preprint arXiv:2310.11511. 





Bi, X.; Chen, D.; Chen, G.; Chen, S.; Dai, D.; Deng, C.; Ding, H.; Dong, K.; Du, Q.; Fu, Z.; et al. 2024. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954. 





Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877– 1901. 





Chen, Y.; Liu, Q.; Zhang, Y.; Sun, W.; Shi, D.; Mao, J.; and Yin, D. 2024. TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy. arXiv preprint arXiv:2406.11678. 





Chen, Y.; Yan, L.; Sun, W.; Ma, X.; Zhang, Y.; Wang, S.; Yin, D.; Yang, Y.; and Mao, J. 2025. Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning. arXiv preprint arXiv:2501.15228. 





Fu, J.; Ng, S.-K.; Jiang, Z.; and Liu, P. 2023. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166. 





Gao, J.; Li, L.; Li, W.; Fu, Y.; and Dai, B. 2024a. SmartRAG: Jointly Learn RAG-Related Tasks From the Environment Feedback. arXiv preprint arXiv:2410.18141. 





Gao, Y.; Xiong, Y.; Wang, M.; and Wang, H. 2024b. Modular rag: Transforming rag systems into lego-like reconfigurable frameworks. arXiv preprint arXiv:2407.21059. 





Ghavamzadeh, M.; Mahadevan, S.; and Makar, R. 2006. Hierarchical multi-agent reinforcement learning. Autonomous Agents and Multi-Agent Systems, 13: 197–229. 





Gong, P.; and Mao, J. 2023. CoAScore: Chain-of-Aspects Prompting for NLG Evaluation. arXiv preprint arXiv:2312.10355. 





Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Vaughan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. 





Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948. 





Hao, S.; Gu, Y.; Ma, H.; Hong, J. J.; Wang, Z.; Wang, D. Z.; and Hu, Z. 2023. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992. 





Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa, A. 2020. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060. 





Huang, J.; and Chang, K. C.-C. 2022. Towards reasoning in large language models: A survey. arXiv preprint arXiv:2212.10403. 





Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Radford, A.; et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276. 





Izacard, G.; Caron, M.; Hosseini, L.; Riedel, S.; Bojanowski, P.; Joulin, A.; and Grave, E. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118. 





Jin, B.; Zeng, H.; Yue, Z.; Yoon, J.; Arik, S.; Wang, D.; Zamani, H.; and Han, J. 2025. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516. 





Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P. S.; Wu, L.; Edunov, S.; Chen, D.; and Yih, W.-t. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In EMNLP (1), 6769–6781. 





Ke, Z.; Kong, W.; Li, C.; Zhang, M.; Mei, Q.; and Bendersky, M. 2024. Bridging the preference gap between retrievers and llms. arXiv preprint arXiv:2401.06954. 





Khattab, O.; Santhanam, K.; Li, X. L.; Hall, D.; Liang, P.; Potts, C.; and Zaharia, M. 2022. Demonstrate-searchpredict: Composing retrieval and language models for knowledge-intensive nlp. arXiv preprint arXiv:2212.14024. 





Khattab, O.; and Zaharia, M. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SI-GIR conference on research and development in Information Retrieval, 39–48. 





Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin, J.; Lee, K.; et al. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7: 453–466. 





Li, X.; Dong, G.; Jin, J.; Zhang, Y.; Zhou, Y.; Zhu, Y.; Zhang, P.; and Dou, Z. 2025. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366. 





Li, X.; Mei, S.; Liu, Z.; Yan, Y.; Wang, S.; Yu, S.; Zeng, Z.; Chen, H.; Yu, G.; Liu, Z.; et al. 2024a. RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards. arXiv preprint arXiv:2410.13509. 





Li, Z.; Wang, J.; Jiang, Z.; Mao, H.; Chen, Z.; Du, J.; Zhang, Y.; Zhang, F.; Zhang, D.; and Liu, Y. 2024b. Dmqrrag: Diverse multi-query rewriting for rag. arXiv preprint arXiv:2411.13154. 





Ma, X.; Gong, Y.; He, P.; Zhao, H.; and Duan, N. 2023. Query rewriting for retrieval-augmented large language models. arXiv preprint arXiv:2305.14283. 





Mallen, A.; Asai, A.; Zhong, V.; Das, R.; Hajishirzi, H.; and Khashabi, D. 2022. When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories. arXiv preprint arXiv:2212.10511, 7. 





Min, S.; Michael, J.; Hajishirzi, H.; and Zettlemoyer, L. 2020. AmbigQA: Answering ambiguous open-domain questions. arXiv preprint arXiv:2004.10645. 





Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744. 





Press, O.; Zhang, M.; Min, S.; Schmidt, L.; Smith, N. A.; and Lewis, M. 2022. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350. 





Robertson, S. E.; and Walker, S. 1994. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In SIGIR’94: Proceedings of the Seventeenth Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, organised by Dublin City University, 232–241. Springer. 





Schulman, J.; Moritz, P.; Levine, S.; Jordan, M.; and Abbeel, P. 2015. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438. 





Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. 





Shao, Z.; Gong, Y.; Shen, Y.; Huang, M.; Duan, N.; and Chen, W. 2023. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. arXiv preprint arXiv:2305.15294. 





Song, H.; Jiang, J.; Min, Y.; Chen, J.; Chen, Z.; Zhao, W. X.; Fang, L.; and Wen, J.-R. 2025. R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2503.05592. 





Su, W.; Tang, Y.; Ai, Q.; Wu, Z.; and Liu, Y. 2024. Dragin: Dynamic retrieval augmented generation based on the real-time information needs of large language models. arXiv preprint arXiv:2403.10081. 





Sun, W.; Yan, L.; Ma, X.; Wang, S.; Ren, P.; Chen, Z.; Yin, D.; and Ren, Z. 2023. Is ChatGPT good at search? investigating large language models as re-ranking agents. arXiv preprint arXiv:2304.09542. 





Sutton, R. S.; Barto, A. G.; et al. 1998. Reinforcement learning: An introduction, volume 1. MIT press Cambridge. 





Team, G.; Anil, R.; Borgeaud, S.; Alayrac, J.-B.; Yu, J.; Soricut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; Millican, K.; et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. 





Team, Q. 2024. Qwen2 technical report. arXiv preprint arXiv:2412.15115. 





Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal, A. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics, 10: 539–554. 





Wang, L.; Yang, N.; Huang, X.; Jiao, B.; Yang, L.; Jiang, D.; Majumder, R.; and Wei, F. 2022. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533. 





Xiao, S.; Liu, Z.; Zhang, P.; Muennighoff, N.; Lian, D.; and Nie, J.-Y. 2024. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, 641–649. 





Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388. 





Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.; Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600. 





Zhang, E.; Wang, X.; Gong, P.; Lin, Y.; and Mao, J. 2024. Usimagent: Large language models for simulating search users. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2687–2692. 





Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou, Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223. 



## Appendix

## A How and why should the cost-based penalty terms be scaled?

We define the cost-based penalty term $R _ { C P }$ in the following Equation (Equal to Equation (4)). 

$$
R _ {C P} = \text { Token } _ {\text { cost }} + \text { Turn } _ {\text { cost }} + \mathbb {I} (S) \tag {11}
$$

We can see that $R _ { C P }$ is composed of three parts: $T o k e n _ { \mathrm { c o s t } } , T u r n _ { \mathrm { c o s t } } ,$ , and I(S). The range of values for these three parts varies significantly. In order to effectively optimize each component, we should normalize their values to approximately between 0 and 1. 

Of course, in real-world scenarios, these three components can all be converted into actual money spent, and each component will have some proportional relationship in terms of cost. However, for the purpose of this study, we consider these three components to be equally important and optimize all three. 

Next, we will introduce how to scale every term in this Equation. 

• $T o k e n _ { \mathrm { c o s t } } { \mathrm { . } }$ : Table 3 presents the average token cost for each executor agent. Among the executable workflows, QR, DS, AG, AS stands out as the most costly, with a token cost of approximately 6.02e-4 dollars per query. By scaling this value to 1.0, we can linearly adjust the token costs of all workflow types, denoted as $T o k e n _ { \mathrm { c o s t } } ,$ to a range between 0 and 1.0. 


Table 3: Agent token cost (per query)


<table><tr><td>Executor Name</td><td>Token Cost</td></tr><tr><td>Query Decomposition Serial (QDS)</td><td>0.91e-4</td></tr><tr><td>Query Decomposition Parallel (QDP)</td><td>1.00e-4</td></tr><tr><td>Query Rewriter (QR)</td><td>0.88e-4</td></tr><tr><td>Document Selector (DS)</td><td>2.08e-4</td></tr><tr><td>Answer Generator (AG)</td><td>1.58e-4</td></tr><tr><td>Answer Summarization (AS)</td><td>1.48e-4</td></tr></table>

• $T u r n _ { \mathrm { c o s t } } { \mathrm { : } }$ This penalty term is related to latency. The more rounds a query requires, the longer it will take to complete, resulting in higher latency. 

Among all the executors, only QDS and QDP incur additional subsequent rounds. Since QDP decomposes the original question into multiple sub-questions that can be searched in parallel, it results in one additional round, as all sub-questions can be processed simultaneously. On the other hand, QDS decomposes the original question into sub-questions that must be searched sequentially, meaning that each subsequent sub-question requires the answer from the previous one to proceed to the next search step. Therefore, QDS results in additional rounds equal to the number of sub-questions. As we limit the maximum number of sub-questions to four rounds, we have normalized the turncost for all executors to a range between 0 and 1, as shown in the table below: 

• I(S): This is a penalty term related to the number of retrieval model calls. Since the cost of calling the search engine’s API is relatively high, this penalty term is included to encourage the planner to minimize the expenses associated with using the search engine. Specifically, if the workflow output by the planner includes a Retrieval Agent (RA), we have ${ \dot { \mathbb { I } } } ( S ) { \dot { = } } 1 { \mathrm { : } }$ ; otherwise, we set $\mathbb { I } ( S ) = 0$ . 


Table 4: $T u r n _ { \mathrm { c o s t } }$ for different workflows:


<table><tr><td>Workflow</td><td>Turn Cost</td></tr><tr><td>QDS</td><td>0.25</td></tr><tr><td>QDP</td><td>0.25, 0.5, 0.75, 1.0</td></tr><tr><td>QR, RA, AG</td><td>0</td></tr><tr><td>RA, DS, AG</td><td>0</td></tr><tr><td>AS</td><td>0</td></tr><tr><td>Other Workflows</td><td>0</td></tr></table>

## B Detailed Values of Different Cost Metrics

In Tables 5, 6, and 7, we present the detailed values of three cost-based metrics—token cost, retrieval call times, and turn number—across various methods on different datasets (averaged per query). Tables 5, 6, and 7 correspond to the three subplots in Figure 3. It is important to note that Figure 3 shows the averages for the NQ and HotpotQA datasets, whereas Tables 5, 6, and 7 present data for all datasets. 

## C Prompt Details

Figures 8 through 13 present the detailed prompt templates used by each agent in the MAO-ARAG framework. Specifically, Figure 8 shows the prompt for the Query Decomposition Serial (QDS) agent. Figure 9 presents the prompt used by the Query Decomposition Parallel (QDP) agent. Figure 10 provides the prompt for the Query Rewriter (QR) agent. Figure 11 illustrates the prompt for the Document Selector (DS) agent. Figure 12 shows the prompt for the Answer Generator (AG) agent. Finally, Figure 13 displays the prompt for the Answer Summarization (AS) agent. 

## D Limitations

Since our MAO-ARAG requires training the planner agent using RL, the planner’s backbone model must effectively follow instructions and have a decent initial ability to plan workflows. In our experiments, it was not possible to train directly based on Qwen2.5-0.5B-Instruct and Qwen2.5- 1.5B-Instruct because models of this size have issues with instruction-following capabilities. 

## E Case Study

To further demonstrate the effectiveness and adaptability of the proposed MAO-ARAG framework, we present a case study illustrating how our system dynamically generates tailored workflows for different types of queries. Each case is structured as follows: we begin with a user query and its corresponding golden answer. Then, we detail each interaction turn within the MAO-ARAG framework. At each turn, the planner selects a workflow by orchestrating a set of executor. The selected workflow is then executed, and the resulting context—including sub-questions, intermediate answers, and retrieved documents—is accumulated. The process continues iteratively until a final answer is produced, at which point it is compared against the golden answer. 

We present four representative cases to highlight MAO-ARAG’s ability to adaptively choose between simple and complex workflows based on query demands. 

## Case 1: Single-Turn Answer Generation Query: Is aluminium a ferrous or non ferrous metal? (From NQ)

This is a straightforward factual question that can be confidently answered from the language model’s internal knowledge. The planner correctly identifies that no retrieval or decomposition is necessary and directly selects the AG module in a single turn. The model produces the correct answer non-ferrous, which matches the golden answer. This case exemplifies MAO-ARAG’s ability to avoid unnecessary computations and costs for simple queries. 

## Single-Turn Answer Generation

• Initial question q: 

– Is aluminium a ferrous or non ferrous metal? 

• Golden answer Ansgolden: 

– non-ferrous 

• Turn 0: 

– Planner: AG 

– Executor: AG 

– Context: 

* Question: Is aluminium a ferrous or non ferrous metal? 

Answer: non ferrous metal 

• Predicted answer Anspredict: 

– non ferrous metal ✓ 

## Case 2: Single-Turn Retrieval-Augmented Generation Query: Who was the editor of the journal Jugantor published in the time of Swadeshi movement? (From NQ)

This question requires external knowledge not reliably stored in the model’s parameters. The planner selects a oneturn plan involving retrieval followed by generation (RA → AG). The retrieved documents contain relevant historical context, enabling the model to correctly identify Bhupendranath Dutt as the editor. Notably, if the model attempted to answer without retrieval, it produced incorrect or hallucinated content. This demonstrates MAO-ARAG’s capability to recognize knowledge gaps and invoke retrieval when necessary. 

## Single-Turn Retrieval-Augmented Generation

• Initial question q: 

– Who was the editor of the journal jugantor published in the time of swadeshi movement? 

• Golden answer Ansgolden: 

– Bhupendranath Dutt 

• Turn 0: 

– Planner: RA, AG 

– Executor: RA 

– Context: 

* Question: Who was the editor of the journal jugantor published in the time of swadeshi movement? 

* Documents: 

· Jugantar Patrika () was a Bengali revolutionary newspaper founded in 1906 in Calcutta by Barindra Kumar Ghosh, Abhinash Bhattacharya and Bhupendranath Dutt. A political weekly, it was founded in March 1906 and served as the propaganda organ for the nascent revolutionary organisation ”Anushilan Samiti”... 

– Executor: AG 

– Context: 

* Question: Who was the editor of the journal jugantor published in the time of swadeshi movement? 

* Documents: 

· Jugantar Patrika () was a Bengali revolutionary newspaper founded in 1906 in Calcutta by Barindra Kumar Ghosh, Abhinash Bhattacharya and Bhupendranath Dutt. A political weekly, it was founded in March 1906 and served as the propaganda organ for the nascent revolutionary organisation ”Anushilan Samiti”... 

Answer: Bhupendranath Dutt 

• Predicted answer Without RAG: 

– Bepin Chandra Pal ✗ 

• Predicted answer With RAG (RA + AG) Anspredict: 

– Bhupendranath Dutt ✓ 

## Case 3: Multi-Turn Parallel Query Decomposition Query: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? (From HotpotQA)

This is a complex comparative question requiring reasoning over multiple independent facts. The planner decomposes the main query into four focused sub-questions: (1) How many members are in the performance act Badly Drawn Boy?, (2) How many instruments are typically used in a performance by Badly Drawn Boy?, (3) How many members are in the performance act Wolf Alice?, and (4) How many instruments are typically used in a performance by Wolf Alice? These sub-queries are processed in parallel, with the planner independently determining whether external retrieval is necessary for each. Once all intermediate results are obtained, the planner invokes the AS module to compute the instrument-to-person ratios and synthesize a comparative answer. This case illustrates MAO-ARAG’s ability to perform fine-grained parallel reasoning and adapt retrieval strategies to the needs of each sub-question. 

## Multi-Turn Parallel Query Decomposition

• Initial question q: 

– Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

• Golden answer Ansgolden: 

– Badly Drawn Boy 

• Turn 0: 

– Planner: QDP 

– Executor: QDP 

– Context: 

* Question: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

* Sub-questions subq: 

· Sub-question 1: How many members are in the performance act Badly Drawn Boy? 

· Sub-question 2: How many instruments are typically used in a performance by Badly Drawn Boy? 

· Sub-question 3: How many members are in the performance act Wolf Alice? 

· Sub-question 4: How many instruments are typically used in a performance by Wolf Alice? 

• Turn 1: 

– planner: AG (parallel for sub-question 1) 

– Executor: AG 

– Context: 

* Question: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

* Sub-questions subq: 

· Sub-question 1: How many members are in the performance act Badly Drawn Boy? 

· Sub-question 2: How many instruments are typically used in a performance by Badly Drawn Boy? 

· Sub-question 3: How many members are in the performance act Wolf Alice? 

· Sub-question 4: How many instruments are typically used in a performance by Wolf Alice? 

* Documents: 

· Documents for Sub-question 1: None 

* Sub-answers suba: 

· Sub-answer 1: One member 

– planner: AG (parallel for sub-question 2) 

– Executor: AG 

– Context: 

Question: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

* Sub-questions subq: 

· Sub-question 1: How many members are in the performance act Badly Drawn Boy? 

· Sub-question 2: How many instruments are typically used in a performance by Badly Drawn Boy? 

· Sub-question 3: How many members are in the performance act Wolf Alice? 

· Sub-question 4: How many instruments are typically used in a performance by Wolf Alice? 

* Documents: 

· Documents for Sub-question 1: None 

· Documents for Sub-question 2: None 

* Sub-answers suba: 

· Sub-answer 1: One member 

· Sub-answer 2: Typically four instruments are used in a performance by Badly Drawn Boy. 

– planner: AG (parallel for sub-question 3) 

– Executor: AG 

– Context: 

* Question: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

* Sub-questions subq: 

· Sub-question 1: How many members are in the performance act Badly Drawn Boy? 

· Sub-question 2: How many instruments are typically used in a performance by Badly Drawn Boy? 

· Sub-question 3: How many members are in the performance act Wolf Alice? 

· Sub-question 4: How many instruments are typically used in a performance by Wolf Alice? 

Documents: 

· Documents for Sub-question 1: None 

· Documents for Sub-question 2: None 

· Documents for Sub-question 3: None 

* Sub-answers suba: 

· Sub-answer 1: One member 

· Sub-answer 2: Typically four instruments are used in a performance by Badly Drawn Boy. 

· Sub-answer 3: Four 

– planner: RA, AG (parallel for sub-question 4) 

– Executor: RA 

– Context: 

* Question: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

* Sub-questions subq: 

· Sub-question 1: How many members are in the performance act Badly Drawn Boy? 

· Sub-question 2: How many instruments are typically used in a performance by Badly Drawn Boy? 

· Sub-question 3: How many members are in the performance act Wolf Alice? 

· Sub-question 4: How many instruments are typically used in a performance by Wolf Alice? 

Documents: 

· Documents for Sub-question 1: None 

· Documents for Sub-question 2: None 

· Documents for Sub-question 3: None 

· Documents for Sub-question 4: construct her musical universe. She created her debut album Coppe entirely on a TEAC Reel- ´ to-Reel, transitioning through Yamaha’s DX7 to the Nord Lead and her current instrument of choice, Teenage Engineering’s versatile OP-1. Outside the synth world Coppe em-´ ploys unconventional instruments including the 5-octave mbira, nail violin and meatgrinder. Coppe characteristically ´ processes her vocals to create a broad range of effects that she likens to “angels whispering,” “colors of the wind,” and “orange sorbet sunsets” with equipment including the Digitech 300 and TC helicon. Coppe co-creates with some of the elec-´ tronic music/IDM scene’s most influential artists. These include Plaid, Kettel,... 

Sub-answers suba: 

· Sub-answer 1: One member 

· Sub-answer 2: Typically four instruments are used in a performance by Badly Drawn Boy. 

· Sub-answer 3: Four 

– Executor: AG 

## – Context:

* Question: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

* Sub-questions subq: 

· Sub-question 1: How many members are in the performance act Badly Drawn Boy? 

· Sub-question 2: How many instruments are typically used in a performance by Badly Drawn Boy? 

· Sub-question 3: How many members are in the performance act Wolf Alice? 

· Sub-question 4: How many instruments are typically used in a performance by Wolf Alice? 

* Documents: 

· Documents for Sub-question 1: None 

· Documents for Sub-question 2: None 

· Documents for Sub-question 3: None 

Documents for Sub-question 4: construct her musical universe. She created her debut album Coppe entirely on a TEAC Reel- ´ to-Reel, transitioning through Yamaha’s DX7 to the Nord Lead and her current instrument of choice, Teenage Engineering’s versatile OP-1. Outside the synth world Coppe em- ´ ploys unconventional instruments including the 5-octave mbira, nail violin and meatgrinder. Coppe characteristically ´ processes her vocals to create a broad range of effects that she likens to “angels whispering,” “colors of the wind,” and “orange sorbet sunsets” with equipment including the Digitech 300 and TC helicon. Coppe co-creates with some of the elec- ´ tronic music/IDM scene’s most influential artists. These include Plaid, Kettel,... 

* Sub-answers suba: 

· Sub-answer 1: One member 

· Sub-answer 2: Typically four instruments are used in a performance by Badly Drawn Boy. 

· Sub-answer 3: Four 

· Sub-answer 4: Four instruments 

• Turn 2: 

– planner: AS 

– Executor: AS 

– Context: 

* Question: Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice? 

* Sub-questions subq: 

· Sub-question 1: How many members are in the performance act Badly Drawn Boy? 

· Sub-question 2: How many instruments are typically used in a performance by Badly Drawn Boy? 

· Sub-question 3: How many members are in the performance act Wolf Alice? 

· Sub-question 4: How many instruments are typically used in a performance by Wolf Alice? 

* Documents: 

· Documents for Sub-question 1: None 

· Documents for Sub-question 2: None 

· Documents for Sub-question 3: None 

Documents for Sub-question 4: construct her musical universe. She created her debut album Coppe entirely on a TEAC Reel- ´ to-Reel, transitioning through Yamaha’s DX7 to the Nord Lead and her current instrument of choice, Teenage Engineering’s versatile OP-1. Outside the synth world Coppe em- ´ ploys unconventional instruments including the 5-octave 

mbira, nail violin and meatgrinder. Coppe characteristically ´ processes her vocals to create a broad range of effects that she likens to “angels whispering,” “colors of the wind,” and “orange sorbet sunsets” with equipment including the Digitech 300 and TC helicon. Coppe co-creates with some of the elec- ´ tronic music/IDM scene’s most influential artists. These include Plaid, Kettel,... 

* Sub-answers suba: 

· Sub-answer 1: One member 

· Sub-answer 2: Typically four instruments are used in a performance by Badly Drawn Boy. 

· Sub-answer 3: Four 

· Sub-answer 4: Four instruments 

* Answer: Badly Drawn Boy 

• Predicted final answer Anspredict: 

– Badly Drawn Boy ✓ 

## Case 4: Multi-Turn Sequential Query Decomposition Query: Ralph Hefferline was a psychology professor at a university that is located in what city? (From HotpotQA)

This is a compositional question requiring sequential reasoning. The planner decomposes the query into two dependent sub-questions: (1) At which university was Ralph Hefferline a psychology professor?, followed by (2) In what city is this university located? The second sub-question is dynamically rewritten based on the first answer. After each planning phase, the appropriate modules are invoked (RA and AG as needed), and the answers are accumulated. Once both sub-questions are resolved, the AS module generates the final answer. This case showcases MAO-ARAG’s ability to handle sequential dependencies through multi-turn planning and sub-question reformulation. 

## Multi-Turn Sequential Query Decomposition

• Initial question q: 

– Ralph Hefferline was a psychology professor at a university that is located in what city? 

• Golden answer Ansgolden: 

– New York City 

• Turn 0: 

– Planner: QDS 

– Executor: QDS 

– Context: 

* Question: Ralph Hefferline was a psychology professor at a university that is located in what city? 

* Sub-questions subq: 

· Sub-question 1: At which university was Ralph Hefferline a psychology professor? 

· Sub-question 2: In what city is this university located? 

• Turn 1: 

– Planner: RA, AG 

– Executor: RA 

– Context: 

* Question: Ralph Hefferline was a psychology professor at a university that is located in what city? 

* Sub-questions subq: 

· Sub-question 1: At which university was Ralph Hefferline a psychology professor? 

· Sub-question 2: In what city is this university located? 

* Documents: 

· Documents for Sub-question 1: of psychology. Ralph Hefferline Ralph Franklin Hefferline (15 February 1910 in Muncie, Indiana – 16 March 1974) was a psychology professor at Columbia University. Hefferline became a patient of Fritz Perls around 1946. He joined a small training group led by Perls in 1948 in New York, and went on to contribute a chapter to the book which defined Gestalt Therapy, ”Gestalt Therapy, Excitement and Growth in the Human Personality”, coauthored by Perls, Paul Goodman and Hefferline, published in 1951. He was the third and junior author and provided the section containing practical exercises. He went on to join the... 

– Executor: AG 

– Context: 

* Question: Ralph Hefferline was a psychology professor at a university that is located in what city? 

* Sub-questions subq: 

· Sub-question 1: At which university was Ralph Hefferline a psychology professor? 

· Sub-question 2: In what city is this university located? 

* Documents: 

· Documents for Sub-question 1: of psychology. Ralph Hefferline Ralph Franklin Hefferline (15 February 1910 in Muncie, Indiana – 16 March 1974) was a psychology professor at Columbia University. Hefferline became a patient of Fritz Perls around 1946. He joined a small training group led by Perls in 1948 in New York, and went on to contribute a chapter to the book which defined Gestalt Therapy, ”Gestalt Therapy, Excitement and Growth in the Human Personality”, coauthored by Perls, Paul Goodman and Hefferline, published in 1951. He was the third and junior author and provided the section containing practical exercises. He went on to join the... 

* Sub-answers suba: 

· Sub-answer 1: Columbia University 

• Turn 2: 

– Planner: AG 

– Executor: AG 

– Context: 

* Question: Ralph Hefferline was a psychology professor at a university that is located in what city? 

* Sub-questions subq: 

· Sub-question 1: At which university was Ralph Hefferline a psychology professor? 

· Sub-question 2: In what city is this university located? 

Documents: 

· Documents for Sub-question 1: of psychology. Ralph Hefferline Ralph Franklin Hefferline (15 February 1910 in Muncie, Indiana – 16 March 1974) was a psychology professor at Columbia University. Hefferline became a patient of Fritz Perls around 1946. He joined a small training group led by Perls in 1948 in New York, and went on to contribute a chapter to the book which defined Gestalt Therapy, ”Gestalt Therapy, Excitement and Growth in the Human Personality”, coauthored by Perls, Paul Goodman and Hefferline, published in 1951. He was the third and junior author and provided the section containing practical exercises. He went on to join the... 

· Documents for Sub-question 2: None 

* Sub-answers suba: 

· Sub-answer 1: Columbia University 

· Sub-answer 1: New York City 

• Turn 3: 

– Planner: AS 

– Executor: AS 

– Context: 

* Question: Ralph Hefferline was a psychology professor at a university that is located in what city? 

* Sub-questions subq: 

· Sub-question 1: At which university was Ralph Hefferline a psychology professor? 

· Sub-question 2: In what city is this university located? 

Documents: 

· Documents for Sub-question 1: of psychology. Ralph Hefferline Ralph Franklin Hefferline (15 February 1910 in Muncie, Indiana – 16 March 1974) was a psychology professor at Columbia University. Hefferline became a patient of Fritz Perls around 1946. He joined a small training group led by Perls in 1948 in New York, and went on to contribute a chapter to the book which defined Gestalt Therapy, ”Gestalt Therapy, Excitement and Growth in the Human Personality”, coauthored by Perls, Paul Goodman and Hefferline, published in 1951. He was the third and junior author and provided the section containing practical exercises. He went on to join the... 

· Documents for Sub-question 2: None 

* Sub-answers suba: 

· Sub-answer 1: Columbia University 

· Sub-answer 1: New York City 

* Answer: New York City 

• Predicted final answer Anspredict: 

– New York City ✓ 


Table 5: Average token cost per query (in milli-USD, mUSD) of various methods across datasets.


<table><tr><td>Methods</td><td>NQ</td><td>PopQA</td><td>AmbigQA</td><td>HotpotQA</td><td>2Wiki</td><td>Musique</td><td>Bamboogle</td><td>Average</td></tr><tr><td>LLM w/o RAG</td><td>0.087</td><td>0.083</td><td>0.085</td><td>0.089</td><td>0.088</td><td>0.088</td><td>0.086</td><td>0.086</td></tr><tr><td>Vanilla RAG</td><td>0.258</td><td>0.265</td><td>0.256</td><td>0.265</td><td>0.271</td><td>0.263</td><td>0.258</td><td>0.262</td></tr><tr><td>RRR (Ma et al. 2023)</td><td>1.160</td><td>1.173</td><td>1.149</td><td>1.172</td><td>1.195</td><td>1.157</td><td>1.107</td><td>1.159</td></tr><tr><td>BGM (Ke et al. 2024)</td><td>0.396</td><td>0.381</td><td>0.386</td><td>0.391</td><td>0.392</td><td>0.379</td><td>0.370</td><td>0.385</td></tr><tr><td>MMOA-RAG (Chen et al. 2025)</td><td>1.709</td><td>1.673</td><td>1.690</td><td>1.688</td><td>1.676</td><td>1.669</td><td>1.594</td><td>1.671</td></tr><tr><td>Self-RAG (Asai et al. 2023)</td><td>1.246</td><td>1.441</td><td>1.215</td><td>1.263</td><td>1.264</td><td>1.085</td><td>0.761</td><td>1.182</td></tr><tr><td>Search-o1 (Li et al. 2025)</td><td>1.535</td><td>1.440</td><td>1.522</td><td>1.362</td><td>1.239</td><td>1.284</td><td>1.093</td><td>1.354</td></tr><tr><td>MAO-ARAG w/o train</td><td>0.162</td><td>0.244</td><td>0.246</td><td>1.124</td><td>1.163</td><td>1.049</td><td>0.674</td><td>0.666</td></tr><tr><td>MAO-ARAG</td><td>0.396</td><td>0.396</td><td>0.387</td><td>1.842</td><td>1.633</td><td>1.735</td><td>1.515</td><td>1.129</td></tr></table>


Table 6: Average number of retrieval call times per query across datasets.


<table><tr><td>Methods</td><td>NQ</td><td>PopQA</td><td>AmbigQA</td><td>HotpotQA</td><td>2Wiki</td><td>Musique</td><td>Bamboogle</td><td>Average</td></tr><tr><td>LLM w/o RAG</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Vanilla RAG</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>RRR (Ma et al. 2023)</td><td>4.0</td><td>4.0</td><td>4.0</td><td>4.0</td><td>4.0</td><td>4.0</td><td>3.936</td><td>3.991</td></tr><tr><td>BGM (Ke et al. 2024)</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>MMOA-RAG (Chen et al. 2025)</td><td>4.0</td><td>4.0</td><td>4.0</td><td>4.0</td><td>4.0</td><td>4.0</td><td>3.936</td><td>3.991</td></tr><tr><td>Self-RAG (Asai et al. 2023)</td><td>0.795</td><td>1.759</td><td>0.747</td><td>1.226</td><td>1.662</td><td>0.898</td><td>0.248</td><td>1.048</td></tr><tr><td>Search-o1 (Li et al. 2025)</td><td>3.929</td><td>3.679</td><td>3.948</td><td>3.518</td><td>3.243</td><td>3.410</td><td>3.064</td><td>3.542</td></tr><tr><td>MAO-ARAG w/o train</td><td>0.410</td><td>0.854</td><td>0.862</td><td>2.712</td><td>2.802</td><td>2.534</td><td>1.864</td><td>1.720</td></tr><tr><td>MAO-ARAG</td><td>1.0</td><td>1.0</td><td>1.0</td><td>3.536</td><td>3.237</td><td>3.413</td><td>3.104</td><td>2.327</td></tr></table>


Table 7: Average number of total turns per query across datasets.


<table><tr><td>Methods</td><td>NQ</td><td>PopQA</td><td>AmbigQA</td><td>HotpotQA</td><td>2Wiki</td><td>Musique</td><td>Bamboogle</td><td>Average</td></tr><tr><td>LLM w/o RAG</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Vanilla RAG</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>RRR (Ma et al. 2023)</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td></tr><tr><td>BGM (Ke et al. 2024)</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>MMOA-RAG (Chen et al. 2025)</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>2.0</td></tr><tr><td>Self-RAG (Asai et al. 2023)</td><td>4.924</td><td>4.675</td><td>4.937</td><td>4.521</td><td>4.238</td><td>4.420</td><td>4.080</td><td>4.542</td></tr><tr><td>Search-o1 (Li et al. 2025)</td><td>4.929</td><td>4.679</td><td>4.948</td><td>4.518</td><td>4.243</td><td>4.410</td><td>4.064</td><td>4.542</td></tr><tr><td>MAO-ARAG w/o train</td><td>1.033</td><td>1.167</td><td>1.090</td><td>2.997</td><td>3.159</td><td>3.014</td><td>2.128</td><td>2.084</td></tr><tr><td>MAO-ARAG</td><td>1.0</td><td>1.0</td><td>1.0</td><td>4.536</td><td>4.237</td><td>4.413</td><td>4.104</td><td>2.899</td></tr></table>


Table 8: The prompt of Query Decomposition Serial Agent.


system: You are a professional assistant skilled at decomposing complex questions into a minimal sequence of logically dependent, independently searchable sub-questions. Each sub-question must: 

- Be self-contained and specific 

- Be suitable for direct information retrieval from search engines or structured databases 

- Be strictly necessary to answer the original question 

You must keep the number of sub-questions as low as possible, and never exceed 4 in total. Avoid redundancy and do not include trivial or overly granular sub-questions. 

assistant: Understood. I will return only factual, retrievable sub-questions, one per line. 

user: Original question is: {content of Question}. 

Now decompose the original question into a logically ordered list of sub-questions. 

Do not number the sub-questions, write one sub-question per line. 

Table 9: The prompt of Query Decomposition Parallel Agent. 

system: You are a professional assistant skilled at decomposing complex multi-entity or multi-location questions into multiple independent and searchable sub-questions. Each sub-question should be specific, logically complete, and not repeat others. assistant: Okay, I will return the parallel sub-questions. 

user: Original question is {content of Question}. 

Break down this question into the minimum number of specific, logically complete, and independently searchable subquestions needed to fully understand and answer the original question. Do not generate more than 4 sub-questions. Each sub-question should be on a separate line, avoid vague demonstratives or repetition, and ensure that each question is self-contained. 

Table 10: The prompt of Query Rewriter Agent. 

system: You are a professional assistant skilled at rewriting overly detailed or redundant questions into a single, concise, and searchable query. Your goal is to keep only the essential part of the question that is needed to find the answer efficiently. 

assistant: Okay, I will return a concise rewritten query. 

user: Original question is: {content of Question}. 

Now rewrite the original question into a single, clear query that focuses only on the essential information needed to find the answer. Avoid unnecessary context, vague references, and maintain specificity. Output only the rewritten query without any extra explanation or formatting. 

Table 11: The prompt of Document Selector Agent. 

system: You are a helpful, respectful and honest assistant. Your task is to output the ID of the candidate Documents (0, 1, 2,..., n) which are helpful in answering the Question. assistant: Okay, I will provide the ID of candidate Documents which are helpful in answering the Question. user: Question is: {content of Question} {content of Documents} assistant: OK, I received the Question and the candidate Documents. user: Now, output the ID of the candidate Documents (0,1,2,...,n) which are helpful in answering the Question: {content of Question}, for example, in the following format: Document0,Document4,Document6,Document7. 

Table 12: The prompt of Answer Generator Agent. 

system: You are a helpful, respectful and honest assistant. Your task is to predict the answer to the question based on the given documents. If you don’t know the answer to a question, please don’t share false information. Answer the question as accurately as possible. 

assistant: Okay, I will provide the answer to the question based on the corresponding documents. Please provide the question and the corresponding documents. 

user: Question is: {content of Question} 

{content of Documents} 

Now, answer the Question: {content of Question}, based on the above Documents. 

assistant: OK, I received the Question and the corresponding Documents. 

user: Given the Question and the corresponding Documents, predict the answer to the Question as briefly and accurately as possible based on the Documents. Only give the brief and accurate answer with the form of **answer**. 

Table 13: The prompt of Answer Summarization Agent. 

system: You are a helpful, respectful and honest assistant. Your task is to predict the final answer to the original question based on the answers to its decomposed sub-questions. If you are not sure about the final answer, do not make up information. Give the most accurate and concise answer possible based on the sub-question answers. 

assistant: Okay, I will provide the final answer to the original question based on the sub-questions and their corresponding answers. Please provide the original question, the sub-questions, and their answers. 

user: Original Question:{content of Question} 

{content of Context} 

Now, based on the above sub-questions and their answers, answer the Original Question: {content of Question} 

assistant: OK, I received the Original Question, its Sub-questions, and their Answers. 

user: Given the Original Question, the Sub-questions and their Answers, predict the final answer to the Original Question as briefly and accurately as possible. Only give the brief and accurate answer in the form of **answer**. 