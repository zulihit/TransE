# TransE: Personal Reproduction

This repository contains a personal reproduction of the NIPS 2013 paper  
**[Translating Embeddings for Modeling Multi-relational Data](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela)**.

---

## Notes
- If you want to train **TransE** on your own dataset, see the `TransE-mydataset.rar` file.  
- The code is relatively old and designed for beginners. It is recommended to first understand the core ideas and basic implementation of TransE here, but not to spend too much time on every implementation detail.  
- For further research, you may want to explore more advanced KGE (Knowledge Graph Embedding) methods, such as:  
  - [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)  
  - [ConvE](https://github.com/TimDettmers/ConvE)  
  - [SelectE](https://github.com/zulihit/SelectE)  

---

## Repository Organization
1. Training and testing code: `src/`  
2. Training and testing results: `res/`  
   - After ~1001 epochs, the loss stabilizes around 14,000 (mostly converged by ~300 epochs).  

---

## Reproducing Results
1. Set the paths for your **DATA** and **save** folders.  
2. Run directly:  

```bash
python transe_simplie.py
```

---

## Reference
- **Paper:** *Translating Embeddings for Modeling Multi-relational Data*  
- **Dataset:** FB15k  

---

## Pseudocode Explanation

![Pseudocode](https://user-images.githubusercontent.com/68625084/166636446-ee7ae1dc-778a-4270-96f6-679868e6d420.png)

**Inputs:**  
- Training triplets  
- Entity set *E*  
- Relation set *L*  
- Margin γ  
- Embedding dimension *k*  

**Steps:**  
1. Initialize relations and entities.  
2. Apply L2 norm normalization to relations.  
3. Entities are initialized without L2 normalization at this step.  
4. Training loop begins:  
   - Normalize entity vectors by L2 norm.  
   - Sample a positive batch (*Sbatch*) of correct triplets.  
   - Construct negative samples by corrupting head/tail entities.  
   - Form training batch (*Tbatch*) with both positive & negative triplets.  
   - Update embeddings using gradient descent.  
5. End training cycle.  

---

## Key Points
- [Zhihu article explanation](https://zhuanlan.zhihu.com/p/508508180?)

---

## Testing
- **isFit**: Choose between `raw` and `filter` evaluation modes.  
  - Note: `filter` mode is significantly slower.  

---

## Example Results (FB15k)

**Training Loss (sample epochs):**
```
epoch: 900  loss: 14122.8202
epoch: 910  loss: 14373.6803
epoch: 920  loss: 14340.6623
epoch: 930  loss: 14373.6773
epoch: 940  loss: 14328.8339
epoch: 950  loss: 14310.5885
epoch: 960  loss: 14262.7636
epoch: 970  loss: 14311.8275
epoch: 980  loss: 14327.8245
epoch: 990  loss: 14146.5392
```

**Evaluation Metrics:**
- Entity hits@10: **0.3077**  
- Entity mean rank: **254.53**  
- Relation hits@10: **0.7907**  
- Relation mean rank: **81.80**  

**Final Results:**  
- Hits@10: **0.4067**  
- Mean Rank: **246.32**  

---

## Acknowledgements
This repo benefits from the following works:  
- [Anery/transE](https://github.com/Anery/transE)  
- [zqhead/TransE](https://github.com/zqhead/TransE)  

Thanks to the authors for their contributions.
