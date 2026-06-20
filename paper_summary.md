[1] A Deep Learning Model for Remaining Useful Life Prediction of Aircraft Turbofan Engine on C-MAPSS Dataset (Asif et al., 2022)
Problem Statement: Accurately estimating the Remaining Useful Life (RUL) of aircraft engines is critical for planning maintenance and avoiding catastrophic failures. A major hurdle in data-driven prognostics is identifying the exact starting point of engine degradation to properly train the models.

Approach & Methods: The authors propose an improved piecewise linear degradation model to determine the onset of deterioration and accurately assign RUL target labels. They pre-process the sensor data using correlation analysis to select only the sensors that show a monotonous degradation trend, followed by a moving median filter to reduce noise. This refined data is then fed into a deep Long Short-Term Memory (LSTM) network.

Results: Validated on the NASA C-MAPSS dataset, the proposed preprocessing and LSTM combination achieved minimum root mean squared error (RMSE) and score function values, outperforming existing baseline methods.

Conclusion: Combining dimensionality reduction and a refined piecewise linear RUL function with deep LSTM networks significantly boosts prediction accuracy for turbofan engines.

[2] Multi-Scale Integrated Deep Self-Attention Network for Predicting Remaining Useful Life of Aero-Engine (Zhao et al., ~2024)
Problem Statement: Engine degradation signatures occur across vastly different temporal scales (e.g., slow long-term wear versus sudden short-term anomalies). Traditional sequential models struggle to capture these multi-scale dependencies simultaneously without losing critical information.

Approach & Methods: The study utilizes a multi-scale integrated architecture heavily relying on self-attention mechanisms (similar to Transformers). This approach allows the network to dynamically weigh the importance of different sensor readings across various time windows, extracting both local degradation features and global, long-range dependencies.

Results: By processing the data at multiple temporal scales, the self-attention network achieves a higher predictive accuracy and handles complex, non-linear degradation trajectories better than standard RNN or CNN architectures.

Conclusion: Multi-scale self-attention networks provide a highly effective way to model the intricate, time-varying relationships within aero-engine sensor data, leading to more robust RUL predictions.

[3] Predictive Maintenance Scheduling for Aircraft Engines Based on Remaining Useful Life Prediction (Wang et al., 2024)
Problem Statement: While predicting RUL is important, translating those complex, continuous predictions into actionable, cost-effective maintenance schedules for airlines remains a significant operational challenge.

Approach & Methods: The authors propose an integrated predictive maintenance framework. They use a hybrid "Trans-LSTM" (Transformer + LSTM) model optimized via Bayesian methods to predict RUL. Crucially, they use these predictions to design an engine alarm threshold. By comparing total flight costs under various scenarios, they calculate the optimal threshold to trigger maintenance tasks.

Results: The data-driven strategy successfully tracks engine status in real-time, drastically reducing the risk of sudden failures compared to standard periodic maintenance while cutting unnecessary maintenance costs.

Conclusion: Linking an advanced Trans-LSTM RUL predictor directly to an optimized economic threshold framework prolongs the engine's optimal operating window and delivers clear financial benefits to airlines.

[4] Physics-Informed Deep Learning Framework for Explainable Remaining Useful Life Prediction (Kim et al., 2025)
Problem Statement: Purely data-driven machine learning models are "black boxes." They often ignore the actual physical mechanics of material degradation (like metal fatigue), making them difficult for aviation engineers to interpret and trust.

Approach & Methods: The study introduces a physics-aware hybrid framework. It synergizes physical Low-Cycle Fatigue (LCF) dynamics—derived from the engine's actual operational stress profiles—with a neural network combining Temporal Convolutional Networks (TCN) and LSTMs.

Results: Tested on a 12-year real-world fleet dataset (F100-PW-229 engines) and the C-MAPSS benchmark, the model reduced RMSE by 29% to 48% compared to purely data-driven state-of-the-art models.

Conclusion: Grounding neural networks in physical fatigue mechanisms not only massively improves predictive accuracy and generalizability to real-world variability but also provides the explainability required for critical aviation decisions.

[5] Remaining Useful Life Prediction of Aero-Engine Enabled by Fusing Knowledge and Deep Learning Models (Li et al., 2023)
Problem Statement: Relying only on raw sensor data ignores the wealth of existing engineering knowledge about how different engine components interact, leaving potential accuracy and interpretability on the table.

Approach & Methods: The researchers propose fusing domain knowledge with deep learning. They represent the physical relationships between sensors as structural flow charts, transforming these charts into embedding vectors. These structural embeddings are then clustered and integrated into the deep learning RUL prediction model alongside the time-series data.

Results: The knowledge-fused approach improved prediction accuracy by 5.5% compared to the best literature baselines.

Conclusion: Injecting structural engineering knowledge into deep learning models enhances both the accuracy and the interpretability of RUL predictions for complex, multi-sensor systems.

[6] FedPM-SGN: A Federated Graph Network for Aviation Equipment Fault Diagnosis by Multisensor Fusion... (Mao et al., 2025)
Problem Statement: Aviation fleets generate data across highly decentralized and heterogeneous sensor configurations. Centralizing this data for fault diagnosis violates privacy, and traditional federated models struggle to handle inputs from differing sensor arrays.

Approach & Methods: The paper introduces a Federated Graph Neural Network (FedPM-SGN). It models multi-sensor data as graphs, where nodes represent sensors and edges represent their relationships. Federated learning allows collaborative training across decentralized nodes, using graph structures to naturally handle variations in sensor setups.

Results: The federated graph framework successfully fused multi-sensor data across heterogeneous clients, achieving higher fault diagnosis accuracy than models trained in isolation or standard non-graph federated networks.

Conclusion: Graph networks deployed in a federated setting offer an elegant, privacy-preserving solution to the sensor heterogeneity problem, capturing spatial-temporal correlations without needing identical input structures across all airlines.

[7] Strategic Integration of Adaptive Sampling and Ensemble Techniques in Federated Learning... (Xu et al., 2025)
Problem Statement: In collaborative federated learning, data quality and the frequency of engine failures vary wildly between airlines. If an airline has mostly healthy data, its weak failure signals get suppressed when averaged with data-rich airlines.

Approach & Methods: The authors design a federated ensemble learning framework based on adaptive sampling. The system combines multiple deep models and uses an adaptive sampling mechanism to prioritize training updates from clients that have underrepresented degradation patterns or lower data quality, ensuring their signals aren't drowned out.

Results: This strategic integration enhanced the global model's robustness and accuracy specifically under heterogeneous (non-IID) data conditions across different distributed clients.

Conclusion: Adaptive sampling combined with ensemble modeling effectively protects weak failure signals and balances the federated learning process across highly unequal airline fleets.

[8] An Efficient Privacy-Preserving Intrusion Detection Scheme for UAV Swarm Networks (Gharami et al., 2025)
Problem Statement: Swarm-based networks (like Unmanned Aerial Vehicles) are highly vulnerable to cyberattacks. Traditional centralized Intrusion Detection Systems (IDS) cause latency and privacy risks, and static models suffer from "model drift" as new attack types emerge.

Approach & Methods: The researchers developed a lightweight, federated continuous learning IDS. They utilize Convolutional Neural Networks (CNN) and LSTMs for local detection. Crucially, they use Elastic Weight Consolidation (EWC) within the federated setup, allowing the model to continuously learn new attack patterns without catastrophically forgetting old ones.

Results: The federated continuous learning model demonstrated exceptional classification accuracies, reaching 99.99% on the UAV-IDS dataset and 98.05% on Cyber-Physical datasets.

Conclusion: Federated continuous learning provides a highly secure, privacy-preserving, and adaptable defense mechanism for decentralized networks, mitigating model drift without sharing raw network data.

[9] Classifier Clustering and Feature Alignment for Federated Learning Under Distributed Concept Drift (Chen et al., 2024)
Problem Statement: Operating conditions shift over time and vary by geography (e.g., Arctic vs. Tropical operations). This causes "distributed concept drift," where the underlying relationship between sensor readings and failure states changes. A single global federated model fails because it tries to average out fundamentally conflicting operational realities.

Approach & Methods: The authors propose "FedCCFA," a framework that clusters client classifiers at the class level rather than forcing a single global model. It aligns the feature spaces of clients that share similar underlying data distributions and adaptively adjusts these alignments based on the entropy of the label distribution.

Results: By grouping clients with similar operational realities, FedCCFA vastly improved generalization performance, reduced gradient conflicts, and maintained high accuracy despite severe data heterogeneity.

Conclusion: Recognizing and clustering different operational profiles (concept drift) rather than punishing differences is vital for building effective federated models in diverse real-world environments.

[10] Federated Learning Framework for Collaborative Remaining Useful Life Prognostics: An Aircraft Engine Case Study (Landau et al., 2026)
Problem Statement: To build robust RUL models, airlines must combine their run-to-failure data. However, strict privacy rules prevent centralization. Furthermore, local sensor data is often corrupted or extremely noisy, which can poison the global federated model.

Approach & Methods: The study implements a collaborative federated learning framework across six simulated airlines using the N-CMAPSS dataset. To handle noise, they introduce four novel, robust parameter aggregation methods (including Softmax and Best-model policies). They also formalize a decentralized validation procedure so airlines can validate the global model without exposing test data.

Results: Federated learning generated more accurate RUL prognostics for five out of the six participating airlines compared to isolated training. The robust aggregation methods successfully immunized the global model against noisy client updates.

Conclusion: Federated learning is a practical and superior alternative to isolated RUL training. Implementing robust aggregation and decentralized validation makes the framework resilient enough for real-world aviation applications.