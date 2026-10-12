# LLM training reliability survey: depth-2 reference closure

This document is the parsed, recursively-expanded reference closure for the
14-paper literature review on LLM training reliability, observability, and
fault diagnosis (the "Nova Vision / ETTR" survey). It is a companion to
[`TELEMETRY_MAPPING.md`](TELEMETRY_MAPPING.md) and [`ARTICLE.md`](ARTICLE.md).

Scope, as requested: **depth-2**. Section 1 resolves each of the 14 seed
papers to a canonical citation. Section 2 is the **transitive closure** --
every work cited by those seed papers, deduplicated and grouped. Section 3
lists the resolution gaps honestly (the one paper whose full bibliography was
not publicly retrievable). Nothing here is fabricated; where a source could not
be reached, it is marked UNRESOLVED and the attempts are stated.

Method: resolution and extraction were done against arXiv, USENIX, DBLP,
OpenAlex, and Semantic Scholar. Where a paper was reachable only as a PDF
(USENIX proceedings, arXiv PDF), the reference list was extracted verbatim with
`pdftotext` and folded in. The private Lark/Feishu wiki that hosts the original
survey could not be fetched (authenticated host), so the 14 seed papers were
taken from the survey text pasted into the conversation.

Coverage update (this pass): bibliographies were fetched and parsed verbatim
from PDF for papers 4 (EROICA), 10 (Aegis), 11 (GREYHOUND), 12 (ResiHP),
13 (R2CCL), and 14 (Llama 3), in addition to papers 1, 2, 3, 5, 6, 8, 9
resolved earlier. All 14 seed papers are now resolved to a canonical citation;
only paper 7 (Intel, SC '25) has a bibliography that is still unretrieved
because its ACM full text is paywalled. See Section 3.

## Contents

1. [The 14 seed papers, resolved](#1-the-14-seed-papers-resolved)
2. [Transitive closure of cited references](#2-transitive-closure-of-cited-references)
3. [Resolution gaps](#3-resolution-gaps)

---

## 1. The 14 seed papers, resolved

| # | Short name | Canonical citation | ID / venue |
| --- | --- | --- | --- |
| 1 | ByteRobust | Wan et al. "Robust LLM Training Infrastructure at ByteDance." SOSP '25. | arXiv:2509.16293 |
| 2 | MegaScale | Jiang et al. "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs." NSDI '24. | arXiv:2402.15627 |
| 3 | ARGUS | Zhou et al. (Tencent). "ARGUS: Production-Scale Tracing and Performance Diagnosis for over 10,000-GPU Clusters." APSys '26. | arXiv:2606.20374 |
| 4 | EROICA | Guan et al. (Alibaba). "EROICA: Online Performance Troubleshooting for Large-scale Model Training." NSDI '26. | arXiv:2506.08528 |
| 5 | Mycroft | Deng et al. (CUHK/ByteDance/Harvard). "Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training." SOSP '25, pp. 254-269. | arXiv:2509.03018 |
| 6 | Revisiting Reliability | Kokolis et al. (Meta FAIR). "Revisiting Reliability in Large-Scale Machine Learning Research Clusters." HPCA '25. | arXiv:2410.21680 |
| 7 | Intel automated failure mgmt | Levitt, Barella, Zeltner, Musta, Cheney, Espinosa, Franza, Gerofi (Intel et al.). "Fine-grained Automated Failure Management for Extreme-Scale GPU Accelerated Systems." SC '25. | DOI 10.1145/3712285.3759883 |
| 8 | Story of Two GPUs | Cui et al. (UIUC/NCSA, Nokia Bell Labs, IBM). "A Story of Two GPUs: A Comparative Study of A100 and H100 GPU Reliability at Scale." SC '25. | arXiv:2503.11901 |
| 9 | Minder | Deng et al. (Tsinghua/ByteDance/Northeastern/Harvard). "Minder: Faulty Machine Detection for Large-scale Distributed Model Training." NSDI '25. | arXiv:2411.01791 |
| 10 | Aegis | Dong et al. (Alibaba Cloud). "Evolution of Aegis: Fault Diagnosis for AI Model Training Service in Production." NSDI '25, pp. 865-881. | USENIX only (not on arXiv) |
| 11 | GREYHOUND | Wu et al. (HKUST/Alibaba). "GREYHOUND: Hunting Fail-Slows in Hybrid-Parallel Training at Scale." USENIX ATC '25, pp. 731-747. | USENIX only (not on arXiv) |
| 12 | ResiHP | Ma et al. (Fudan/Shanghai AI Lab/HKUST/SJTU/CUHK). "ResiHP: Taming LLM Training Failures with Dynamic Hybrid Parallelism." | arXiv:2605.06374 |
| 13 | R2CCL | Wang et al. (Univ. of Maryland). "Reliable and Resilient Collective Communication Library for LLM Training and Serving." | arXiv:2512.25059 |
| 14 | Llama 3 | Llama Team, Meta AI (Grattafiori, Dubey, et al.). "The Llama 3 Herd of Models." Meta AI technical report, 2024. | arXiv:2407.21783 |

Notes:
- Papers 3 (ARGUS) and 12 (ResiHP) carry future-dated arXiv IDs (`2606.*`,
  `2605.*`) as returned by the resolver in this environment; venues are as
  listed.
- Papers 10, 11 are USENIX proceedings-only. Papers 10 (Aegis, `nsdi25-dong.pdf`)
  and 11 (GREYHOUND, `atc25-wu-tianyuan.pdf`) were fetched and their reference
  lists extracted verbatim. Paper 4 (EROICA) is on arXiv after all
  (arXiv:2506.08528, NSDI '26); its bibliography [1]-[40] was fetched. Paper 7
  (Intel, SC '25) resolved via DBLP to a DOI-only ACM paper; its full text is
  paywalled so its bibliography could not be fetched.

---

## 2. Transitive closure of cited references

Every distinct work cited by at least one of the 14 seed papers, deduplicated
across bibliographies and grouped by role. The **Cited by** column lists the
seed papers (by the numbers in Section 1) whose bibliography contains the
entry; it is a lower bound (limited to the bibliographies retrieved --
papers 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14; only paper 7's
paywalled bibliography is missing).
"Named tools" at the end covers non-paper artifacts (profilers,
libraries, docs) that the survey and its sources lean on.

### 2.1 Telemetry, diagnosis, and fault detection

| Work | Cited by |
| --- | --- |
| py-spy sampling profiler (Frederickson) | 1, 3 |
| NVIDIA CUPTI (CUDA Profiling Tools Interface) | 3, 5 |
| NVIDIA DCGM / DCGM Diagnostics | 9 |
| NVIDIA EUD (Extended Utility Diagnostics) | 1, 9 |
| NVIDIA Nsight Systems | 3, 5 |
| PyTorch Profiler | 3, 5, 9 |
| PyTorch Flight Recorder (prototype, stuck-job debugging) | 6, 14 |
| Holistic Trace Analysis (HTA), Meta Research | 3 |
| Chakra: standardized execution traces (Sridharan et al. 2023) | 5 |
| Mystique: production AI benchmark generation (Liang et al. 2022) | 5 |
| Neutrino (Huang and Wu, OSDI '25) | 3 |
| XPUTimer (Cui et al. 2025) | 5 |
| Holmes: localizing irregularities in LLM training (Yao et al. 2025) | 3 |
| Unicron: economizing self-healing LLM training (He et al. 2023/2024) | 6, 9, 12 |
| C4: communication-driven anomaly detection (Dong et al. 2024/2025, HPCA) | 1, 3, 5 |
| SuperBench: proactive validation (Xiong et al., ATC '24) | 1, 4, 9 |
| Aegis fault diagnosis (Dong et al., NSDI '25) | 1, 3, 4 |
| Minder faulty-machine detection (Deng et al., NSDI '25) | 3, 5 |
| GREYHOUND fail-slow hunting (Wu et al., ATC '25) | 3, 5 |
| EROICA online troubleshooting (Guan et al., NSDI '26) | 3 |
| Mycroft collective-comm tracing (Deng et al., SOSP '25) | 3, 4 |
| Characterization of LLM development in the datacenter (Hu et al., NSDI '24) | 3, 5, 6, 9 |
| Fail-slow at scale (Gunawi et al., FAST '18) | 5, 8 |
| Gray Failure: Achilles' heel of cloud-scale systems (Huang et al., HotOS '17) | 5 |
| Dr. DNA: combating SDC via neuron activation distributions (Ma et al., ASPLOS '24) | 6 |
| Silent Data Corruptions at Scale (Dixit et al. 2021) | 1, 6 |
| Cores that don't count (Hochschild et al. 2021) | 1 |
| Understanding/mitigating hardware failures in DL training (He et al., ISCA '23) | 6 |
| Understanding GPU errors on large-scale HPC (Tiwari et al., HPCA '15) | 6 |
| Memory errors over the lifetime of Cielo (Levy et al., SC '18) | 6 |

### 2.2 Distributed tracing and network diagnosis

| Work | Cited by |
| --- | --- |
| Dapper: large-scale distributed tracing (Google, 2010) | 5 |
| X-Trace: pervasive network tracing (Fonseca et al., NSDI '07) | 5 |
| Pivot Tracing (Mace et al., SOSP '15 / ATC '18) | 5 |
| The Benefit of Hindsight: tracing edge-cases (NSDI '23) | 5 |
| Pingmesh: DC network latency measurement (Guo et al., SIGCOMM '15) | 2, 5, 9, 10 |
| R-Pingmesh: service-aware RoCE monitoring (Liu et al., SIGCOMM '24) | 4, 5, 9, 10 |
| EverFlow / packet-level telemetry (Zhu et al., SIGCOMM '15) | 2, 10 |
| LossRadar: fast lost-packet detection (Li et al., ACM Conf. on emerging Networking EXperiments and Technologies 2016) | 2, 10 |
| NetBouncer: link failure localization (Tan et al., NSDI '19) | 2, 9, 10 |
| Hostping: intra-host RDMA bottlenecks (Liu et al., NSDI '23) | 2, 5, 9, 10 |
| Collie: RDMA subsystem anomalies (Kong et al., NSDI '22) | 9, 10 |
| SIMON: sensing/inference in DC networks (Geng et al., NSDI '19) | 9 |
| NetAssistant: dialogue-based network diagnosis (Wang et al., NSDI '24) | 9 |
| ExChain: exception dependency analysis (Li et al., NSDI '24) | 5 |
| Fathom: datacenter application network performance (Qureshi et al., SIGCOMM '23) | 9, 10 |
| Murphy: performance diagnosis of distributed cloud apps (Harsh et al., SIGCOMM '23) | 9, 10 |
| Scouts: domain-customized incident routing (Gao et al., SIGCOMM '20) | 9 |
| Reverse traceroute (Katz-Bassett et al., NSDI '10) | 9 |
| PathDump: datacenter network debugging (Tammana et al., OSDI '16) | 9, 10 |
| Gestalt: unified fault localization (Mysore et al., ATC '14) | 9 |
| FChain: black-box online fault localization (Nguyen et al., ICDCS '13) | 9 |
| FluxRank: localizing root-cause machines (Liu et al., ISSRE '19) | 9 |
| Deepview: virtual hard disk failure localization | 9 |
| Alibaba HPN: DC network for LLM training (Qian et al., SIGCOMM '24) | 4, 6, 10 |
| VL2: scalable and flexible data center network (Greenberg et al., SIGCOMM '09) | 4 |
| 007: democratically finding the cause of packet drops (Arzani et al., NSDI '18) | 10 |
| NetPoirot: taking the blame game out of DC operations (Arzani et al., SIGCOMM '16) | 10 |
| Vigil: closing the network diagnostics gap (Arzani et al., SIGCOMM '17) | 10 |
| PINT: probabilistic in-band network telemetry (Ben Basat et al., SIGCOMM '20) | 10 |
| Trumpet: timely/precise triggers in DCs (Moshref et al., SIGCOMM '16) | 10 |
| Confluo: monitoring/diagnosis for high-speed networks (Khandelwal et al., NSDI '19) | 10 |
| SpiderMon: closed-loop network diagnosis (Wang et al., NSDI '22) | 10 |
| Zeno: performance diagnosis with temporal provenance (Wu et al., NSDI '19) | 10 |
| G2: graph processing for diagnosing distributed systems (Guo et al., ATC '11) | 10 |
| DeepFlow: network-centric distributed tracing (Shen et al., SIGCOMM '23) | 10 |
| LOF: density-based local outliers (Breunig et al., SIGMOD '00) | 10 |
| Passive realtime datacenter fault detection (Roy et al., NSDI '17) | 10 |
| Isolation Forest anomaly detection | 10 |

### 2.3 Anomaly detection, statistics, and ML-for-systems

| Work | Cited by |
| --- | --- |
| OmniAnomaly: stochastic RNN for multivariate TS (Su et al., KDD '19) | 9, 10 |
| Donut: VAE anomaly detection for seasonal KPIs (Xu et al., WWW '18) | 9 |
| Opprentice: ML-based anomaly detection (Liu et al., IMC '15) | 9 |
| EGADS: automated time-series anomaly detection (Laptev et al., KDD '15) | 9 |
| VAE-LSTM hybrid anomaly detection (Lin et al., ICASSP '20) | 9 |
| CTF: coarse-to-fine anomaly transfer (Sun et al., INFOCOM '21) | 9, 10 |
| Illuminating the gray zone: gray-failure localization in server OS (Zhang et al., FSE '24) | 11, 12 |
| Robust/rapid KPI clustering (Li et al., IWQoS '18) | 9 |
| Symbolic Pattern Forest TS clustering (Li et al., IJCAI '19) | 9 |
| Change-point analysis of multivariate data (Matteson and James, JASA '14) | 9 |
| Bayesian online changepoint detection (Agudelo et al. 2020) | 11 |
| DBSCAN: density-based clustering with noise (Ester et al., KDD '96) | 4 |
| HDBSCAN: hierarchical density-based clustering (McInnes et al., JOSS '17) | 4 |
| Mean shift: robust feature-space analysis (Comaniciu and Meer, TPAMI '02) | 4 |
| Pattern Recognition and Machine Learning (Bishop and Nasrabadi 2006) | 4 |
| Network Time Protocol / internet time synchronization (Mills, IEEE Trans. Comm. '91) | 4 |
| Sage: ML-driven microservice performance debugging (Gan et al., ASPLOS '21) | 9 |
| Interpreting DL-based networking systems (Meng et al., SIGCOMM '20) | 9 |
| Survey of black-box model explanation (Guidotti et al., CSUR '18) | 9 |
| Random forest feature contribution (Palczewska et al., IRI '13) | 9 |
| Top-down first-order logical decision trees (Blockeel and De Raedt, AIJ '98) | 9 |
| Mahalanobis distance / robust variants (Mahalanobis; Ghorbani; Leys et al.) | 9 |
| Pearson / Kendall / Spearman correlation (Cohen et al.; Kendall '38; Spearman) | 9 |
| z-score transformation (Cheadle et al. 2003) | 9 |
| Distance/similarity measures survey (Weller-Fahy et al. 2014) | 9 |
| Bandwidth-optimal all-reduce (Patarasuk and Yuan, JPDC '09) | 9 |
| Jensen's operator inequality (Hansen and Pedersen 2002) | 6 |
| Slowdown scheduling convergence (Harchol-Balter et al. 2002) | 6 |
| BSP bridging model (Valiant, CACM '90) | 6 |

### 2.4 Fault tolerance, elasticity, and recovery

| Work | Cited by |
| --- | --- |
| Bamboo: preemptible-instance resilience (Thorpe et al., NSDI '22) | 1, 11, 12 |
| Oobleck: pipeline-template resilience (Jang et al., SOSP '23) | 1, 11, 12 |
| Parcae: proactive parallelism reconfiguration (Duan et al., NSDI '24) | 1, 11 |
| Varuna: low-cost elastic training (Athlur et al., EuroSys '22) | 1, 12 |
| Malleus: straggler-resilient hybrid parallel training (Li et al., SIGMOD '24) | 5 |
| KungFu: adaptive distributed ML (Mai et al., OSDI '20) | 5 |
| CPR: partial-recovery fault tolerance for DLRM (Maeng et al., MLSys '20) | 6 |
| Erasure-coded fault tolerance for recommendation (Zhang et al., VLDB '23) | 6 |
| Proactive fault tolerance via task migration (Chakravorty et al. 2005/2006) | 6 |
| Ekko: fail-slow avoidance in parameter servers (Sima et al. 2022) | 1 |
| GPU age-aware scheduling (Zimmer et al., SC '18) | 6 |
| DiLoCo: distributed low-communication training (Douillard et al. 2023) | 6 |
| Recycle: resilient training via pipeline adaptation (Gandhi et al., SOSP '24) | 12, 13 |
| Adaptra: straggler-resilient hybrid-parallel training (Wu et al. 2025) | 12 |
| ElasWave: elastic-native scalable hybrid-parallel training (Kang et al. 2025) | 12 |
| TrainMover: interruption-resilient ML training runtime (Lao et al. 2024) | 13 |
| Just-in-time checkpointing: low-cost error recovery (Gupta et al., EuroSys '24) | 13 |
| Swift: expedited failure recovery for large-scale DNN training (Zhong et al., PPoPP '23) | 11 |
| Whale: efficient giant model training over heterogeneous GPUs (Jia et al., ATC '22) | 11 |
| HetPipe: large DNN training on heterogeneous GPU clusters (Park et al., ATC '20) | 11 |
| Metis: fast automatic distributed training on heterogeneous GPUs (Um et al., ATC '24) | 11 |
| PERSEUS: fail-slow detection for cloud storage (Lu et al., FAST '23) | 11, 12 |
| IASO: fail-slow detection/mitigation for distributed storage (Panda et al., ATC '19) | 12 |
| Understanding/detecting fail-slow hardware bugs in cloud systems (Dong et al., ATC '25) | 12 |
| FT2: first-token-inspired online fault tolerance for LLMs (Sun et al., HPDC '25) | 12 |
| Portus: efficient DNN checkpointing to persistent memory (Li et al., ICDCS '24) | 11 |
| Understanding, detecting, localizing partial failures (Lou et al., NSDI '20) | 11 |
| Capturing/enhancing in-situ system observability (Huang et al., OSDI '18) | 11 |

### 2.5 Checkpointing

| Work | Cited by |
| --- | --- |
| GEMINI: in-memory checkpoints (Wang et al., SOSP '23) | 1, 5, 6, 11, 12, 13 |
| CheckFreq: frequent fine-grained checkpointing (Mohan et al., FAST '21) | 1, 5, 11, 12, 13 |
| Check-N-Run: differential/quantized checkpoints (Eisenman et al. 2022) | 1, 13 |
| ByteCheckpoint: parallelism-agnostic checkpoints (Wan et al. 2025) | 1, 8, 13 |
| Survey of checkpointing strategies / Young-Daly (Bautista-Gomez et al. 2024) | 6 |
| Higher-order optimum checkpoint interval (Daly, FGCS '06) | 6 |
| First-order optimum checkpoint interval (Young, CACM '74) | 6 |

### 2.6 Parallelism and training systems

| Work | Cited by |
| --- | --- |
| Megatron-LM (Shoeybi et al. 2019) | 2, 3, 4, 5, 6, 9, 11, 12, 13, 14 |
| Efficient large-scale training on GPU clusters w/ Megatron (Narayanan et al., SC '21) | 2, 3, 5, 12, 14 |
| PipeDream: generalized pipeline parallelism (Narayanan et al., SOSP '19) | 2, 3, 5, 12 |
| GPipe: pipeline parallelism (Huang et al., NeurIPS '19) | 2, 3, 5, 12, 14 |
| ZeRO: memory optimizations (Rajbhandari et al., SC '20) | 2, 3, 5, 11, 12, 14 |
| PyTorch FSDP (Zhao et al., VLDB '23) | 2, 3, 5, 12, 14 |
| Breadth-first pipeline parallelism (Lamy-Poirier, MLSys '23) | 14 |
| PyTorch Distributed / DDP (Li et al. 2020) | 2, 9 |
| PyTorch (Paszke et al., NeurIPS '19) | 2 |
| PyTorch 2 / TorchDynamo (Ansel et al., ASPLOS '24) | 6 |
| TensorFlow (Abadi et al., OSDI '16) | 2 |
| Alpa: automating inter/intra-operator parallelism (Zheng et al., OSDI '22) | 3, 5 |
| GShard (Lepikhin et al. 2020) | 2, 13 |
| Switch Transformers (Fedus et al., JMLR '22) | 3 |
| Reducing activation recomputation (Korthikanti et al., MLSys '23) | 2, 14 |
| FlashAttention (Dao et al., NeurIPS '22) | 3 |
| FlashAttention-2 (Dao 2023) | 2 |
| FlashAttention-4 (Dao-AILab 2025) | 3 |
| Zero Bubble pipeline parallelism (Qi et al., ICLR '24) | 5 |
| Comm-computation overlap via decomposition (ASPLOS '22) | 5 |
| Megatron-Turing NLG 530B (Smith et al. 2022) | 2, 9 |
| Pathways: async distributed dataflow (Barham et al., MLSys '22) | 1, 6 |
| MAST: global ML scheduling (Choudhury et al., OSDI '24) | 6, 14 |
| Borg: cluster management at Google (Verma et al., EuroSys '15) | 6 |
| SLURM (Yoo et al., JSSPP '03) | 6, 8 |
| Optimus: dynamic DL scheduler (Peng et al., EuroSys '18) | 2 |
| Analysis of multi-tenant GPU clusters (Jeon et al., ATC '19) | 5, 6 |
| Project Adam (Chilimbi et al., OSDI '14) | 6 |
| Large scale distributed deep networks (Dean et al., NeurIPS '12) | 6 |
| Hogwild! lock-free SGD (Recht et al., NeurIPS '11) | 6 |
| Mixed precision training (Micikevicius et al., ICLR '18) | 2 |
| Communication-efficient parameter server (Li et al., NeurIPS '14) | 9 |
| Comm scheduling: P3, TicTac, ByteScheduler, Pipe-SGD, SAPipe, preemptive all-reduce | 2 |
| Orca: serving system for generative models (OSDI '22) | 5 |
| Congestion control: DCQCN (Zhu et al. '15), Swift (Kumar et al. '20), PFC 802.1Qbb | 2 |
| RDMA over Ethernet (RoCE) for distributed AI training at Meta scale (Gangidi et al., SIGCOMM '24) | 14 |
| Holistic configuration management at Facebook (Tang et al., SOSP '15) | 14 |
| Open MPI (Graham et al. 2005) | 5 |
| Rail-only network for trillion-param LLMs (Wang et al., HOTI '23) | 6 |
| Optimized network architectures for LLM training (arXiv) | 6 |
| Is Network the Bottleneck of Distributed Training? (Zhang et al., NetAI '20) | 5 |
| DL training in Facebook data centers (Naumov et al. 2020) | 5 |
| MAD-Max distributed acceleration (Hsia et al., ISCA '23) | 6 |
| TPU v4 optically-reconfigurable supercomputer (Jouppi et al., ISCA '23) | 6 |
| In-datacenter TPU performance analysis (Jouppi et al., ISCA '17) | 6 |
| AI-enabling workloads on large GPU systems (Li et al., HPCA '22) | 6 |
| IBM Gen AI infrastructure (Gershon et al. 2024) | 6 |

### 2.7 Models, LLMs, and capabilities (scene-setting tail)

| Work | Cited by |
| --- | --- |
| Llama 3 Herd of Models (Dubey/Grattafiori et al. 2024) | 3, 5, 8; self=14 |
| LLaMA / Llama 2 (Touvron et al. 2023) | 14 |
| GPT-4 technical report (OpenAI/Achiam et al. 2023) | 2, 9, 11, 14 |
| GPT-3 / few-shot learners (Brown et al., NeurIPS '20) | 2, 3 |
| GPT-2 / unsupervised multitask learners (Radford et al. 2019) | 9 |
| Scaling laws for neural LMs (Kaplan et al. 2020) | 2, 3 |
| PaLM (Chowdhery et al. 2022/2023) | 2, 3, 6, 9 |
| BLOOM (Scao et al. 2022) | 2, 3 |
| OPT (Zhang et al. 2022) | 2 |
| Attention is all you need (Vaswani et al., NeurIPS '17) | 9, 14 |
| Segment Anything (Kirillov et al., ICCV '23) | 6 |
| Emu image generation (Dai et al. 2023) | 6 |
| Audiobox audio generation (Vyas et al. 2023) | 6 |
| Deep residual learning / ResNet (He et al., CVPR '16) | 2 |
| PEFT / quantization: LoRA, SmoothQuant, GPTQ, LLM.int8() | 2 |
| Efficient attention: Longformer, Sparse Transformers, linear attention, RWKV, RetNet | 2 |
| Instruction-tuned / chat models: Alpaca, Vicuna, Koala, BELLE, FLAN | 2 |
| QSGD gradient quantization (Alistarh et al., NeurIPS '17) | 2 |
| Unsupervised translation of programming languages (Lachaux et al. 2020) | 6 |

### 2.8 Named tools and non-paper artifacts

Monitoring/observability stack referenced across the seed papers and the
survey: **Prometheus**, **Grafana**, **Perfetto**, **Vector** (Datadog),
**NCCL Profiling Kit (NPKit)**, **GVProf**, **NCCL-tests**, **NCCL** and
**NCCLX** (Meta), **torch.distributed**, **NDTimeline** (survey's in-house
framework-semantics tracer), **Onenet / NCCL Analysis (Covas)** and
**Lagrange / Malphite / Gazer** (survey's in-house platforms), NVIDIA **XID
error messages**, NVIDIA **System Management Interface (nvidia-smi)**, **dmesg**,
**Intel Performance Counter Monitor (PCM)**, and the **SHIELD** self-healing
interconnect. EROICA (paper 4) additionally leans on **NVIDIA Nsight Compute**,
**CUPTI**, **NVML**, **DCGM**, **PyTorch Profiler**, **Kineto**, **Dynolog**
(Meta), **NCCL** and the **NCCL profiler plugin**, **Gloo**, **JAX**,
**Kubernetes**, **SLURM**, **NVIDIA NeMo**, **bpftrace / eBPF**, and
**Mellanox mstflint**. Hardware/platform docs: NVIDIA A100/DGX-A100, DGX-2H,
GB200 NVL72, ConnectX-6 Dx / ConnectX-7, H100/H800, NVLink/NVSwitch, PCIe,
IPMI v2.0, TPU v5p / AI Hypercomputer.

---

## 3. Resolution gaps

Stated plainly so the closure is not mistaken for complete:

- **Paper 7 (Intel automated failure management): citation resolved,
  bibliography not retrieved.** Resolved via DBLP to Levitt, Barella, Zeltner,
  Musta, Cheney, Espinosa, Franza, and Gerofi, "Fine-grained Automated Failure
  Management for Extreme-Scale GPU Accelerated Systems," SC '25
  (DOI 10.1145/3712285.3759883). Franza and Gerofi are Intel, matching the
  survey's "by Intel". It is not on arXiv and the ACM full text is paywalled
  (403 to automated fetch), so its reference list could not be extracted. This
  is the only seed paper whose bibliography remains unretrieved.
- **Paper 4 (EROICA): bibliography now retrieved.** It is on arXiv after all
  (arXiv:2506.08528v4, NSDI '26); the earlier assumption that it was USENIX-
  proceedings-only was wrong. Its references [1]-[40] were extracted from the
  arXiv PDF and folded into Section 2 (DBSCAN, HDBSCAN, mean shift, Bishop
  pattern recognition, NTP, VL2, R-Pingmesh, Alibaba HPN, SuperBench, plus the
  seed papers Mycroft/Aegis/MegaScale/Megatron and its tooling stack).
- **Paper 14 (Llama 3): bibliography now retrieved** from the arXiv PDF text
  layer (200+ references). Its list is LLM/eval-heavy; only the training-
  infrastructure subset (Megatron-LM, ZeRO, GPipe, PyTorch FSDP, activation
  recomputation, RoCE at Meta scale, breadth-first pipeline parallelism, MAST,
  holistic configuration management) is folded into Section 2.
- **Paper 11 (GREYHOUND): bibliography now retrieved verbatim** as entries
  [1]-[65] from the USENIX ATC '25 PDF (`atc25-wu-tianyuan.pdf`), superseding
  the earlier bib-key mnemonics; folded into Section 2.
- **Paper 10 (Aegis): bibliography now retrieved verbatim** as entries [1]-[64]
  from the USENIX NSDI '25 PDF (`nsdi25-dong.pdf`); network-diagnosis and
  anomaly-detection heavy, folded into Sections 2.2 and 2.3.
- **Paper 12 (ResiHP): bibliography retrieved** as entries [1]-[56] from arXiv;
  the M-Z tail warrants light verification.
- **Paper 13 (R2CCL): bibliography retrieved** (~40 author-year entries) from
  arXiv. Reference to Zhu et al. 2024 (vehicle-to-vehicle) is a genuine entry
  in the R2CCL bibliography, not a mis-link.
- **Paper 9 (Minder): complete [1]-[82]**, except verbatim author/venue text
  for [80]-[82] (inline descriptors captured: Spearman correlation, cloud
  diagnosis, Deepview).
- **Depth-2 boundary.** The **Cited by** counts are a lower bound: they reflect
  the bibliographies retrieved (papers 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13,
  and 14). References unique to the unretrieved bibliography of paper 7 are
  necessarily missing from the closure.

The closure was intentionally stopped at depth-2 (seed papers plus their direct
references). Expanding to depth-3 (the references' references) would multiply
the list several-fold and is out of the requested scope.
