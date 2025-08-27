**VTR-LLM**
VTR-LLM is a system that brings the power of Large Language Models (LLMs) to FPGA computer-aided design (CAD) tools. It integrates LLMs with the VTR (Versatile Place and Route) toolchain to provide natural-language assistance during CAD tool usage and development.

This repository contains both:

* Baseline VPR-LLM – which queries LLMs with detailed natural language prompts about VPR internals and logs.

* RAG-VPR-LLM – which augments the LLM with a retrieval-augmented generation (RAG) system that improves performance, reduces costs, and enables scaling to large toolchains.

**How to use:**
To run a single case use `vpr_llm.py` and to run the all the testcases run `launch_full_testcases.py` 

**Contribution guidelines**
To contribure a testcase that you faced, please open a pull request adding the testcase under `testcases` directory. The testcase should include:
1. circuit file
2. architecture file
3. VPR log where the error happens
4. README file explaining the issue and proposing the best solution if you know it
5. (optional) any extra files used in the vpr command
