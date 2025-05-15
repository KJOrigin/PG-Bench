# CCPG Benchmark (PG-Bench) & KEGA 
<p float="left"><img src="https://img.shields.io/badge/python-v3.9+-red"> <img src="https://img.shields.io/badge/pytorch-v2.6+-blue">

This repository serves as an open-source repository for the **CCPG benchmark (PG-Bench) and KEGA**, housing open-source datasets, model weights, and source code, and is the implementation of the paper "Advancing Conversation-based Chinese Patient Guidance with A New Benchmark and Knowledge-Evolvable Guidance Assistant".

<img src="Pictures/figure1.png" alt="figure1" border="0">


## 📂 Dataset Overview
<img src="Pictures/table1.png" alt="table1" border="0">

### PG-Bench Dataset
- **General.jsonl** - General Hospital Patient-Doctor Dialogue Guidance Dataset
- **Gynecological.jsonl** - Gynecological Specialty Hospital Patient-Doctor Dialogue Guidance Dataset
- **Pediatric.jsonl** - Pediatric Specialty Hospital Patient-Doctor Dialogue Guidance Dataset  
- **Stomatological.jsonl** - Stomatological Specialty Hospital Patient-Doctor Dialogue Guidance Dataset
- **TCM.jsonl** - TCM Specialty Hospital Patient-Doctor Dialogue Guidance Dataset

## 🧠 KEGA Architecture
<img src="Pictures/figure2.png" alt="figure2" border="0">

## 📊 Benchmark Results
For more detailed benchmark results, please [Click here]()
<img src="Pictures/table2.png" alt="table2" border="0">

## ✨ KEGA Performance
For more detailed performance results, please [Click here]()
<img src="Pictures/table3.png" alt="table3" border="0">
<img src="Pictures/table4.png" alt="table4" border="0">
<img src="Pictures/table5.png" alt="table5" border="0">

## 📖 Usage
You can implement our methods according to the following steps:

1. Install the necessary packages. Run the command:
   ```
   pip install -r requirements.txt
   ```
2. Install Swift to deploy models. Please [Click here](https://swift.readthedocs.io/zh-cn/latest/index.html)
3. Run our code using Python.
   
   Train the KEGA
   ```
   python KEGA_train.py
   ```
   Evaluate the KEGA
   ```
   python KEGA_test.py
   ```
   Zero-Shot Testing
   ```
   python zeroshot.py
   ```
   Few-Shot Testing
   ```
   python fewshot.py
   ```

## 🌟 Contributions and suggestions are welcome!
