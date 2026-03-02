# docs/overall.md, Efficient Multi-Agent Memory System (EMAMS) 项目整体结构设计


├── assets/                        # 相关资源，如提示词模板、图示等
├── benchmarks/                     # 评测基准和数据集
|   ├── FrontierScience/
|   └── HLE-Verified/
├── docs/                           # 文档和设计说明
├── experiments/                     # 实验脚本和配置
│   ├── logs/                            # 运行日志
│   ├── results/                         # 实验结果和分析
│   ├── scripts/                         # 各类实用脚本，如数据处理、评测等
├── src/
|   |
│   ├── common/                      # 跨层共享的数据结构
│   │   ├── message.py               # MASMessage, AgentMessage, TaskResult
│   │   └── __init__.py
|   |
│   ├── llm/
│   │   ├── base.py
│   │   └── openai.py
│   │
│   ├── mas/
│   │   ├── autogen/
│   │   │   ├── __init__.py
│   │   │   ├── autogen_prompt.py
│   │   │   └── autogen.py
│   │   ├── dylan/
│   │   │   ├── __init__.py
│   │   │   ├── dylan_prompt.py               # DylanMAS(MetaMAS)
│   │   │   ├── dylan.py
│   │   │   └── prompt.py
│   │   ├── macnet/
│   │   │   ├── __init__.py
│   │   │   ├── graph_mas.py               # MacNetMAS(MetaMAS)
│   │   │   ├── graph_prompt.py
│   │   │   ├── graph.py             # 图拓扑逻辑
│   │   │   └── node.py
│   │   ├── reasoning/
|   |   |   ├── __init__.py
|   |   |   └── base.py
│   │   ├── __init__.py              # MAS 相关的抽象和接口设计
│   │   ├── format.py                # 任务输入输出的格式规范
│   │   ├── base.py                  # MetaMAS 抽象类（唯一需要精心设计的接口）
│   │   │
│   ├── memory/                      # 核心研究对象，精心设计
│   │   ├── base.py                  # MASMemoryBase 抽象
│   │   ├── storage/
│   │   │   ├── base.py
│   │   │   ├── vector_store.py
│   │   │   └── graph_store.py
│   │   ├── retrieval/
│   │   │   ├── base.py
│   │   │   └── ...
│   │   └── methods/
│   │       ├── empty.py
│   │       ├── generative.py
│   │       ├── voyager.py
│   │       ├── g_memory.py
│   │       └── ours/
│   │
│   │
│   ├── configs/
│   ├── benchmarks/
│   ├── experiments/
│   └── results/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt