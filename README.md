<h1 align="center">联邦推荐系统</h1>

> 联邦推荐系统主要是为了能够在保证用户隐私的前提下，学习到一个个性化同时还能够运用已有知识的推荐系统

## Contents-目录



## 从这里开始

这个Repository记录我从零开始学习入门联邦推荐系统的过程，如果有任何问题或者交友学术合作等欢迎大家联系我：[邮箱](mailto:liweidengdavid@gmail.com)，[个人主页](https://liweidengdavid.github.io/)。

## 建议关注的公众号

1. 智能之道（每日推送arxiv上新发表的联邦学习和推荐系统的相关论文）
2. 机器学习与推荐算法（每周推送推荐系统的研究进展）
3. 智荐阁（会推送推荐系统相关文章）

## 论文阅读

### Survey

1. [A Survey on Federated Recommendation Systems (2025)](https://ieeexplore.ieee.org/abstract/document/10423793)
   - 联邦推荐系统的入门综述，全面讲述了联邦推荐系统。
     - 内容：联邦推荐系统的定义，面临的挑战，常见的通信结构，隐私保护的方法，常见的攻击方式和防卫方法，异质性 (heterogeneity)，减少通信消耗的方法，联邦推荐系统的应用，公共的Benchmark Datasets和未来的研究方向。
2. [A Survey of Recommendation Systems: Recommendation Models, Techniques, and Application Fields (2022)](https://www.mdpi.com/2079-9292/11/1/141)
   - 推荐系统的入门综述（适合推荐系统的零基础小白，如果对推荐系统有一定的了解，可以跳过这一篇文章。）
     - 内容：全面讲述了推荐技术，应用场景和未来方向。
3. [A Survey on Large Language Models for Recommendation (2024)](https://arxiv.org/pdf/2305.19860)
   - LLM在Recommendation System（RS）的综述
     - 内容：讲述了LLMs与RS结合（LLM4Rec）的相关技术和现有工作重点是利用LLMs强大的Zero-shot/Few-shot 能力，并且也说明了LLM4Rec的挑战与困难
   
4. [When Foundation Model Meets Federated Learning: Motivations, Challenges, and Future Directions (2024)](https://arxiv.org/pdf/2306.15546)
   - 联邦学习（Federated Learning，）与基础模型（Foundation Model，FM）的综述
     - 内容：讲述了为什么要将FL技术和FM结合起来？FL技术可以帮助FM解决什么问题？FM解决FL中的什么问题？将FL和FM结合起来的挑战？未来的研究方向？


### 具体的联邦推荐系统的算法

1. [FedFast: Going Beyond Average for Faster Training of Federated Recommender Systems(2020)](https://dl.acm.org/doi/pdf/10.1145/3394486.3403176)
2. [Efficient-FedRec: Efficient Federated Learning Framework for Privacy-Preserving News Recommendation (2021)](https://arxiv.org/pdf/2109.05446)
3. [FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling (2022)](https://dl.acm.org/doi/pdf/10.1145/3534678.3539119)
4. [Multi‑center federated learning: clients clustering for better  personalization (2022)](https://link.springer.com/content/pdf/10.1007/s11280-022-01046-x.pdf)
5. 
7. [FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data(2016)](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
   - ***Contribution***：这篇文章是联邦学习的**开山鼻祖**，其开创了联邦学习这个领域，证明了联邦学习的有效性。***Implementation***：本文使用的方法很简单，就是根据每一个客户端的数据量的多少来作为对应的权重，进而去聚合来自不同客户端的模型，进而在server端得到了一个global model，最终就是每一个client都使用这个global model。
7. [Secure Federated Matrix Factorization(2020)](https://arxiv.org/pdf/1906.05108)
   - 本文提出了一个名为FedMF (Federated Matrix Factorization)的模型。***Contribution***：本文证明了如果server能够得到同一个user连续两轮的上传的item梯度，那么server就可以反推得到该user对于该item的评分。***Motivation：*** 因为本文证明了之前的联邦学习方法没办法完全保证用户隐私，仍然有机会泄露用户的隐私信息。***Implementation***：因此采用了homomorphic encryption同态加密算法（公钥私钥）对上传的梯度进行加密，使得能够保护用户的rating信息不泄露。
   - ***Tips：*** 由于FedMF采用了异步（asynchronous）更新的策略（一旦server收到client上传的信息就更新全局参数），因此其不需要类似FedAvg这样的聚合Function，其聚合过程在optimizer中完成。***Advances：***可以解决部分client中途退出的问题。***Problems：*** Staleness（陈旧性）problems（可能收到的更新信息是陈旧的，因为部分的client已经更新了多次，部分的client才更新一次）影响模型的收敛。
8. [Federated Neural Collaborative Filtering(2021)](https://arxiv.org/pdf/2106.04405v1)
   - 本文提出了一个名称为FedNCF（Federated Neural Collaborative Filtering）的Framework。***Contribution***：将SecAvg protocol引入到FedAvg中解决传统存在的隐私泄露风险的问题，提出了名称为MF-SecAvg的聚合方法。***Motivation：***（1）之前的联邦学习方法只假设了Server是honest-but-curious，但是没有考虑Client是honest-but-curious；（2）之前的MF只能刻画user-item的线性关系，复杂的关系没办法刻画；（3）FedAvg运用到MF中会导致模型性能下降。（因为当更新item embedding时候，只有和当前client交互过的item的embedding会发生改变，未交互的是不会有改变，直接运用FedAvg会导致模型收敛变慢，效果差）
9. [FedGNN: Federated Graph Neural Network for Privacy-Preserving Recommendation(2021)](https://arxiv.org/pdf/2102.04925)
   - 本文提出了一个名称为FedGNN的模型。***Contribution：***（1）提出了一种在保护用户隐私前提下的user-item graph expansion方式，进而能够构建user-item的高阶关联信息。（2）同时提出了将LDP（local differential privacy）和pseudo item采样技术结合起来保护与user交互过的item。***Motivation：***（1）Federated Learning将client数据集独立存储，但是现有的工作证明挖掘user-item高阶信息是十分重要的，而独立存储只能包含user-item一阶信息，那么如何在保证user隐私的前提下挖掘高阶信息是本文想解决的问题。（2）以往的方法（如FedAvg等FL方法）只能保护user-item rating信息，server可以通过item的非零梯度推断出与该user交互的item是什么。***Implementation***：For（1）会对item id和user id进行同态加密，加密只有方式只有中央server知道。这里会引入一个第三方服务器（可以是不可信的），其只需要做一件事，就是对加密后的item id进行match，如果match，就交换加密后的user id，两个user就匹配成为了邻居；For（2）FedGNN使用了LDP和pseudo item采样技术结合起来保护与user交互过的item。
10. [Federated Social Recommendation with Graph Neural Network(2021)](https://dl.acm.org/doi/pdf/10.1145/3501815)
    - 本文提出了一个名称为FeSoG（Fedrated Social recommendation with Graph neural network），是第一个将social information纳入FedRec中的工作。
11. [Dual personalization on federated recommendation (2023)](https://arxiv.org/pdf/2301.08143)
       - 本文提出了名为Personalized Federated Recommendation（PFedRec）方法。***Contribution***：item embedding和score embedding（代替user embedding）是personalization。***Motivation***：不同user看待item视角不同，因此item embedding应该也是personalization的。***Implementation***：Score Function是完全personalization，不会发送到中央server中进行聚合；Item  Embedding其是会被send to server中进行聚合学习global item view，然后将其作为每一个client的个性化item embedding的init，然后在使用client私有化数据进行finetuning。
11. [Federated Recommendation with Additive Personalization (2024)](https://arxiv.org/pdf/2301.09109)
       - 本文提出了名称为Federated Recommendation with Additive Personalization (FedRAP)的方法。其实际上是在PFedRec的方法基础上进行的改进。***Contribution:*** 将一个item embedding分成了两个部分user-specific item embedding（save in the local client）和global item embedding（send to 中央服务器进行聚合）。***Motivation:*** 虽然不同的user看待item的视角不同（user-specific item embedding），但是也有共通的部分（global item embedding）。***Implementation***：item embedding=user-specific item embedding+global item embedding。同时为了通信效率和表征的有效性，本文提出了两个regularization term，（1）为了保证通信效率，要求Global item embedding是稀疏的。（2）为了保证表征有效性，user-specific item embedding和global item embedding应该要尽可能不同，二者应该呈现互补的样式。
11. [GPFedRec: Graph-Guided Personalization for Federated Recommendation (2024)](https://arxiv.org/pdf/2305.07866)
    - 本文提出了名称为Graph-Guided Personalization for Federated Recommendation（GPFedRec:）方法。这是一种新型的聚合方式（FedAVG）的改进版本。***Contribution:*** 其提出了一种新型的聚合方式，其是基于user-relation graph来聚合。其核心点在于：如何在Federated Learning的架构下，构建user-relation graph并且不增加泄露user隐私的风险。***Motivation:*** 在集中式的推荐系统中，其他的工作已经证明将user-relation graph引入推荐系统可以提高系统的推荐正确率，但是如何在联邦学习的框架下得到user-relation graph，并且不损害user隐私。***Implementation***：通过对没一个client上传的item embedding进行cos相似度计算，进而得到user-relation graph。后续通过user-relation graph去引导item embedding聚合等。
12. [Personalized Federated Collaborative Filtering: AVariational AutoEncoder Approach(2024)](https://arxiv.org/pdf/2408.08931)
    - 本文提出了一个新的personalized Federated Collaborative Filtering （FedCF） method, which incorporates a gating dual-encoder VAE, named FedDAE。***Contribution:*** 其提出了通过VAE去建模user item之间的复杂关系，而不是受限于user embedding and item embedding vector，进而能够捕捉更加复杂的非线性关系。***Motivation:*** 现有的大部分推荐系统模型都是基于矩阵分解的思路，由于矩阵分解思路只能捕捉模型的线性关系，无法捕捉非线性的复杂关系。虽然如今采用了Neural Collaborative Filtering (NCF)的方式来，但是仍然首先于item and user embedding vector的形式，同时现有的大部分NCF都rely on personalized item embedding，因此导致模型的泛化性能差。***Implementation***：我们通过构建dual-encoder VAE（一个用来捕捉global信息，一个用来捕捉local信息），然后通过gate网络来自适应的调节二者（global and local）的权重，同时还保证的模型的泛化性能。
13. [When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User Interactions (2024)](https://arxiv.org/pdf/2305.12650)
    - 本文提出了一个名称为Item-aligned Federated aggregation framework for cold-start Recommendation (IFedRec)的模型。***Contribution:***其是在联邦学习的设置下第一个处理cold-start recommendation的模型，其通过不增加数据泄露风险的方式引入item attribute信息来应对cold-start问题。***Motivation：***现有的FedRec没有关注冷启动问题，但是这个是一个实实在在存在的问题，而且是推荐系统中的难点。***Implementation***：其分成两个阶段：（1）Learning on the Warm Items；（2）Inference on the Cold Items。（1）*Stage1：*通过在server端（item attribute一般也存储在server端，因此不会增加数据泄露的风险）引入一个Meta attribute Network，根据item属性信息来得到对应的item attribute embedding（这个在center推荐系统中被证明是一个解决cold-start问题的好方法）。其还在server和client端各自引入了一个item representation alignment mechanism来训练Meta attribute Network和为client端的item embedding补充来自item attribute的信息。（2）*Stage2：*当遇到Cold-start items时，首先server的Meta attribute Network计算得到item attribute embedding，每一个client在下载到本地上，将其作为Item embedding，与user embedding一起送入到Rating Prediction中得到预测分数，进行推荐。
    - ***Tips：***其这里还使用了**alternative update method**的小训练Trick，使得模型在cold-start中有着不错的表现。
    
14. [Federated Adaptation for Foundation Model-based Recommendations (2024)](https://arxiv.org/pdf/2405.04840)
    - 本文提出了一个Federated recommendation with Personalized Adapter (FedPA)的模型。***Contribution:***其是在联邦学习设置下第一个将推荐系统和Foundation Model结合起来的模型。***Motivation：***由于Foundation Model (FM) 是在真实世界的大数据集中pre-train来的，因此其内部包含有common knowledge，这个对于Recommendation System来说是重要的。但是将FM与推荐系统结合起来，特别是在联邦学习的设置下，具有以下两个**Challenges**：（1）client端计算资源和存储空间有限，因此常见的FM没办法在Client端部署。（2）如何将FM中的common knowledge和personalization knowledge合理融合起来。***Implementation***：For Challengs (1) 本文采用知识蒸馏knowledge distillation (KD)方法将FM蒸馏出一个小模型进行使得client端也能够运行和存储对应的模型。For Challenges (2) 本文采用Adaptive Gate Learning Mechanism来自适应的得到Common Knowledge and personalization Knowledge（User-Level Personalization and User-Group-Level Personalization（具体采用类似LoRA方式的Personalized Adapter））的权重。

15. 



## 数学基础

1. 机器学习

   - 书本：[统计学习方法（第2版)](https://drive.google.com/file/d/1ArqwIOF4Mydl-3gP9FtxEoonPlR_P8b3/view?usp=sharing)
   - 进阶版书本：[模式识别与机器学习](https://drive.google.com/file/d/1Za9VSiUuizQGUWHwxYvOaA4C3xXiaHDJ/view?usp=sharing)

2. 深度学习

   - 书本：[花书](https://drive.google.com/file/d/12_TjoxIxnI9ME7_UoC2hEyrBm2BEq3HO/view?usp=sharing)
   - 视频：[李沐大神-动手学深度学习](https://space.bilibili.com/1567748478/lists/358497?type=series)

3. 矩阵分析

   - 视频：[《Matrix Methods in Data Analysis, Signal Processing, and Machine Learning | Mathematics | MIT OpenCourseWare》](https://www.bilibili.com/video/BV1b4411j7V3/?vd_source=f017e2676c557e3448fe0a3320c88af8)

4. 凸优化

   - 书本：[凸优化](https://drive.google.com/file/d/1KPe6CX1rXEzoHv2pGIlfzkkbUs8BnXoS/view)
   - 视频：[凸优化书本作者斯坦福公开课](https://www.youtube.com/watch?v=1menqhfNzzo&list=PLoROMvodv4rMJqxxviPa4AmDClvcbHi6h)

   





