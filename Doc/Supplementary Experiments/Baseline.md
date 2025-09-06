**GPT-3.5-turbo/GPT-4o**: Developed by OpenAI, GPT-3.5-turbo is an optimized GPT-3.5 variant for efficient inference with strong natural language understanding and generation capabilities. GPT-4o, the latest in the GPT series, advances reasoning and contextual understanding, setting a high benchmark for complex NLP tasks. Both models were accessed via OpenAI’s API for our experiments.

**DeepSeek-V3/DeepSeek-R1**: DeepSeek-V3 is a general-purpose language model excelling in text generation, summarization, and question answering, with strong generalization and contextual understanding. DeepSeek-R1, optimized for multi-step reasoning, is pre-trained on synthetic chain-of-thought data and fine-tuned via reinforcement learning. It is released in both full and distilled versions. We evaluated DeepSeek-V3 and DeepSeek-R1-Distill-Qwen-32B via commercial APIs.

**Qwen2/2.5**: Developed by Alibaba Cloud, Qwen2 is a general-purpose large language model for tasks like text generation, summarization, and question answering, achieving strong performance on multilingual and domain-specific benchmarks. Qwen2.5, its upgrade, enhances Qwen2's architecture and training, improving accuracy, coherence, and efficiency, particularly in complex tasks such as long-context understanding and low-resource language processing.

**Qwen3**: Developed by Alibaba Cloud, Qwen3 is a family of advanced large language models designed for exceptional performance across text generation, coding, mathematics, and general tasks. Building on the Qwen2.5 foundation, Qwen3 introduces hybrid thinking modes for flexible reasoning, supports 119 languages, and offers multiple model sizes from 600 million to 235 billion parameters, with open-sourced weights available.

**GLM4-9B**: Developed by Zhipu AI, is the open-source iteration of the latest generation in the GLM-4 series of pretrained models.

**Baichuan2-7B**: Developed by Baichuan Intelligence, Baichuan2-7B is a new-generation open-source large language model trained on 2.6 trillion tokens of high-quality data. It achieves advanced performance on multiple authoritative benchmarks for Chinese, English, and multilingual tasks, excelling in both general and domain-specific applications.

**Mistral-7B-Instruct-v0.3**: Developed by Mistral AI, Mistral-7B-Instruct-v0.3 is an instruction-tuned version of the Mistral-7B model, designed for tasks like text generation and summarization. It demonstrates competitive performance on benchmarks, while maintaining efficiency with advanced architectural optimizations.

**Llama-3.1-8B-Instruct**: Developed by Meta AI, Llama-3.1-8B-Instruct is an instruction-tuned model designed for natural language understanding and generation tasks, such as text generation and question answering. It features 8 billion parameters and demonstrates strong performance on various benchmarks.

**Yi-1.5-9B-Chat**: Developed by Zero One All Things Technology,Yi-1.5-9B-Chat is a large-scale conversational model with 9 billion parameters, designed for interactive dialogue generation and natural language understanding. It demonstrates strong performance in dialogue-based tasks, achieving competitive results in conversational AI benchmarks.

**Gemma2-9B-instruct**: Developed by Google, Gemma2-9B-Instruct is a lightweight, open LLM optimized for instruction-tuned text generation. With 9B parameters, it offers efficient deployment and strong performance in tasks like summarization and QA.

**BianCang-7B**: Developed by the Qilu University of Technology (Shandong Academy of Sciences) and the Shandong University of Traditional Chinese Medicine, BianCang-7B is a large language model specifically designed for the Traditional Chinese Medicine (TCM) domain. It is based on Qwen2/2.5 architecture and trained using a two-stage process that incorporates TCM and modern medical knowledge. BianCang-7B excels in TCM diagnosis, treatment, and medical exam tasks, outperforming existing open-source models in specialized areas.

**BioMistral-7B**: Developed by a collaboration between Avignon University, Nantes University, BioMistral-7B is an open-source large language model tailored for the biomedical domain. It is based on the Mistral-7B-Instruct-v0.1 model and further pre-trained on PubMed Central data. BioMistral-7B demonstrates superior performance on medical question-answering tasks.

**Taiyi-7B**: Developed by the DUTIR lab, Taiyi-7B is a bilingual large language model fine-tuned for diverse biomedical tasks. It excels in intelligent biomedical question-answering, doctor-patient dialogues, report generation, information extraction, machine translation, headline generation, and text classification.

**Llama-3.1-8B-UltraMedical**: Developed by Tsinghua C3I Lab, this 8B model is fine-tuned for biomedical tasks, achieving excellent scores among 7B-level models on medical benchmarks.

**Llama3-OpenBioLLM-8B**: Developed by Saama AI Labs, this 8-billion-parameter model is fine-tuned for biomedical applications, including clinical note summarization, medical question answering, and entity recognition.

**Lingdan-13B**: Developed by TCMAI-BJTU, Lingdan-13B is a large language model specifically designed for traditional Chinese medicine (TCM) applications. It builds upon the Baichuan2-13B architecture and is trained on extensive TCM datasets, including ancient texts, pharmacopoeias, and clinical records. The model excels in clinical reasoning tasks, such as diagnosis and prescription recommendation.

**Sunsimiao-7B**: Developed by X-D Lab, Sunsimiao-7B is a Chinese medical large language model designed to provide accurate medical consultations and diagnostic support. It is fine-tuned on extensive Chinese medical datasets and achieves excellent performance on the CMB-Exam benchmark, with notable accuracy in national medical licensing exams.

**HuatuoGPT-o1-7B**: Developed by The Chinese University of Hong Kong and  Shenzhen Research Institute of Big Data, based on Qwen2.5-7B. It excels in complex medical reasoning by generating detailed thought processes before providing answers. The model outperforms both general-purpose and medical-specific LLMs in benchmarks, showcasing advanced reasoning capabilities.

**WiNGPT2-Gemma-2-9B-Chat**: Developed by Winning Health. It offers human-like AI doctor consultations and general medical knowledge Q&A for the public, while providing diagnostic, medication, and health advice suggestions for healthcare professionals.