# Dataset Preprocessing README


## 🔥 June 2, 2025 Update for TUAB and TUEV

The previous preprocessing code for the **TUAB** and **TUEV** datasets was inherited from the [BIOT](https://github.com/ycq091044/BIOT) and [LaBraM](https://github.com/935963004/LaBraM) repositories. These original implementations included **random elements** in the data splitting process. Even with fixed random seeds, different hardware environments could lead to **inconsistent Train/Val/Test splits**. This issue has been carried forward into **CBraMod**.

In the performance comparison presented in the **CBraMod paper**, I directly cited the results reported in the **BIOT** and **LaBraM** papers without having access to their exact dataset splits. Therefore, I cannot guarantee that the comparisons were made using the **same dataset partitions**. As a result, the evaluation may **not be entirely fair**.

Moreover, others may also be unable to reproduce a **fair comparison** with **CBraMod** on **TUAB** and **TUEV** under the same conditions.

To fully address this issue, I have updated the preprocessing code for **TUAB** and **TUEV** to ensure **fixed, deterministic dataset splits**. If you are conducting experiments on these two datasets, please use the **latest version of the preprocessing code** to generate the splits.

For accurate and fair comparisons, it is **strongly recommended** to re-implement existing methods such as **BIOT**, **LaBraM**, and **CBraMod** **on the same fixed splits**.

### 📊 Current Sample Counts (Updated Preprocessing)

#### **TUAB:**
- **Train:** 67,436  
- **Validation:** 15,634  
- **Test:** 29,421  
- **Total:** 112,491  

#### **TUEV:**
- **Train:** 297,103  
- **Validation:** 75,407  
- **Test:** 36,945  
- **Total:** 409,455