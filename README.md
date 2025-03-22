<h1 align="center">联邦推荐系统</h1>

## Contents-目录



## 从这里开始

这个repository记录我从零开始学习入门联邦推荐系统的过程，如果有任何问题或者交友学术合作等欢迎大家联系我：[邮箱](mailto:liweidengdavid@gmail.com)，[个人主页](https://liweidengdavid.github.io/)

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
3. [Federated Neural Collaborative Filtering(2021)](https://arxiv.org/pdf/2106.04405)
4. [FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling (2022)](https://dl.acm.org/doi/pdf/10.1145/3534678.3539119)
5. [Multi‑center federated learning: clients clustering for better  personalization (2022)](https://link.springer.com/content/pdf/10.1007/s11280-022-01046-x.pdf)
6. 
7. [FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data(2016)](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
   - 这篇文章是联邦学习的开山鼻祖，其开创了联邦学习这个领域，证明了联邦学习的有效性。本文使用的方法很简单，就是根据每一个客户端的数据量的多少来作为对应的权重，进而去聚合来自不同客户端的模型，进而在server端得到了一个global model，最终就是每一个client都使用这个global model。
8. [Federated Neural Collaborative Filtering(2021)](https://arxiv.org/pdf/2106.04405v1)
9. [Dual personalization on federated recommendation (2023)](https://arxiv.org/pdf/2301.08143)
   - 本文提出了名为Personalized Federated Recommendation（PFedRec）方法。***Contribution***：item embedding和score embedding（代替user embedding）是personalization。***Motivation***：不同user看待item视角不同，因此item embedding应该也是personalization的。***Implementation***：Score Function是完全personalization，不会发送到中央server中进行聚合；Item  Embedding其是会被send to server中进行聚合学习global item view，然后将其作为每一个client的个性化item embedding的init，然后在使用client私有化数据进行finetuning。
10. [Federated Recommendation with Additive Personalization (2024)](https://arxiv.org/pdf/2301.09109)
    - 本文提出了名称为Federated Recommendation with Additive Personalization (FedRAP)的方法。其实际上是在PFedRec的方法基础上进行的改进。***Contribution:*** 将一个item embedding分成了两个部分user-specific item embedding（save in the local client）和global item embedding（send to 中央服务器进行聚合）。***Motivation:*** 虽然不同的user看待item的视角不同（user-specific item embedding），但是也有共通的部分（global item embedding）。***Implementation***：item embedding=user-specific item embedding+global item embedding。同时为了通信效率和表征的有效性，本文提出了两个regularization term，（1）为了保证通信效率，要求Global item embedding是稀疏的。（2）为了保证表征有效性，user-specific item embedding和global item embedding应该要尽可能不同，二者应该呈现互补的样式。
11. [GPFedRec: Graph-Guided Personalization for Federated Recommendation (2024)](https://arxiv.org/pdf/2305.07866)
    - 本文提出了名称为Graph-Guided Personalization for Federated Recommendation（GPFedRec:）方法。这是一种新型的聚合方式（FedAVG）的改进版本。***Contribution:*** 其提出了一种新型的聚合方式，其是基于user-relation graph来聚合。其核心点在于：如何在Federated Learning的架构下，构建user-relation graph并且不增加泄露user隐私的风险。***Motivation:*** 在集中式的推荐系统中，其他的工作已经证明将user-relation graph引入推荐系统可以提高系统的推荐正确率，但是如何在联邦学习的框架下得到user-relation graph，并且不损害user隐私。***Implementation***：通过对没一个client上传的item embedding进行cos相似度计算，进而得到user-relation graph。后续通过user-relation graph去引导item embedding聚合等。
12. [Personalized Federated Collaborative Filtering: AVariational AutoEncoder Approach(2024)](https://arxiv.org/pdf/2408.08931)
    - 本文提出了一个新的personalized Federated Collaborative Filtering （FedCF） method, which incorporates a gating dual-encoder VAE, named FedDAE。***Contribution:*** 其提出了通过VAE去建模user item之间的复杂关系，而不是受限于user embedding and item embedding vector，进而能够捕捉更加复杂的非线性关系。***Motivation:*** 现有的大部分推荐系统模型都是基于矩阵分解的思路，由于矩阵分解思路只能捕捉模型的线性关系，无法捕捉非线性的复杂关系。虽然如今采用了Neural Collaborative Filtering (NCF)的方式来，但是仍然首先于item and user embedding vector的形式，同时现有的大部分NCF都rely on personalized item embedding，因此导致模型的泛化性能差。***Implementation***：我们通过构建dual-encoder VAE（一个用来捕捉global信息，一个用来捕捉local信息），然后通过gate网络来自适应的调节二者（global and local）的权重，同时还保证的模型的泛化性能。
13. [When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User Interactions (2024)](https://arxiv.org/pdf/2305.12650)
    - 
14. [Federated Adaptation for Foundation Model-based Recommendations (2024)](https://arxiv.org/pdf/2405.04840)
    - 
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

   





