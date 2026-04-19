# Awesome AI Agents [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of AI agent frameworks, tools, platforms, research papers, and resources.

AI Agents are autonomous systems that use LLMs to reason, plan, and take actions. This list tracks the rapidly evolving ecosystem.

**Contributing:** PRs welcome! Read the [contribution guidelines](CONTRIBUTING.md) first.

---

## Contents

- [Frameworks & Libraries](#frameworks--libraries)
- [Platforms & Low-Code](#platforms--low-code)
- [Agent Infrastructure](#agent-infrastructure)
- [Evaluation & Testing](#evaluation--testing)
- [Safety & Governance](#safety--governance)
- [Research Papers](#research-papers)
- [Tutorials & Courses](#tutorials--courses)
- [Use Cases & Case Studies](#use-cases--case-studies)
- [Community](#community)

---

## Frameworks & Libraries

### Multi-Agent Orchestration
- [AG2](https://github.com/ag2ai/ag2) — Successor to AutoGen. Multi-agent framework with improved APIs.
- [Agent Swarm](https://github.com/desplega-ai/agent-swarm) — Multi-agent orchestration for AI coding assistants (Claude Code, Codex, Gemini CLI). Lead/worker coordination with Docker isolation, compounding memory, and Slack/GitHub integration.
- [AgentField](https://github.com/Agent-Field/agentfield) — Open-source control plane that makes AI agents callable as microservices. Routing, coordination, memory, async execution, and cryptographic audit trails. Supports Python, Go, and TypeScript.
- [AgentScope](https://github.com/agentscope-ai/agentscope) — Alibaba's production-ready agent framework with essential abstractions, built-in fine-tuning support, and a visual drag-and-drop interface.
- [Agno](https://github.com/agno-agi/agno) — Programming language for agentic software. Build and manage multi-agent systems at scale.
- [AutoGen](https://github.com/microsoft/autogen) — Microsoft's multi-agent conversation framework. Supports complex agent topologies.
- [CAMEL](https://github.com/camel-ai/camel) — Communicative agents for role-playing and multi-agent cooperation. First LLM multi-agent framework.
- [CrewAI](https://github.com/crewAIInc/crewAI) — Role-based multi-agent framework. Agents with roles, goals, and backstories.
- [DeerFlow](https://github.com/bytedance/deer-flow) — ByteDance's open-source long-horizon SuperAgent harness. Orchestrates sub-agents, sandboxes, memory, tools, and skills for tasks spanning minutes to hours. Hit #1 GitHub Trending with v2.0 (Feb 2026).
- [dimos](https://github.com/dimensionalOS/dimos) — Agentic operating system for physical space. Build multi-agent systems that control humanoids, quadrupeds, drones, and other hardware via natural language.
- [Google Agent Development Kit (ADK)](https://github.com/google/adk-python) — Google's open-source, code-first Python framework for building multi-agent systems with A2A support.
- [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful agent workflows as graphs. Part of the LangChain ecosystem.
- [Mastra](https://github.com/mastra-ai/mastra) — TypeScript-first AI agent framework with workflows, RAG, and integrations.
- [MetaGPT](https://github.com/geekan/MetaGPT) — Multi-agent framework that mimics a software company with roles (PM, architect, engineer).
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) — Framework for building, orchestrating and deploying multi-agent workflows (Python + .NET).
- [MiroFish](https://github.com/666ghj/MiroFish) — Concise and universal swarm intelligence engine for forecasting and prediction. Upload seed material, describe goals in natural language, get a detailed prediction report and an interactive simulation.
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) — OpenAI's production framework for multi-agent orchestration with handoffs and guardrails.
- [Ruflo](https://github.com/ruvnet/ruflo) — Agent orchestration platform optimized for Claude. Features self-learning swarms, distributed intelligence, RAG integration, and native Claude Code/Codex integration. Formerly claude-flow.
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) — Microsoft's SDK for AI orchestration. Plugins, planners, and memory.
- [Swarm](https://github.com/openai/swarm) — OpenAI's lightweight multi-agent framework (educational).

### Single Agent
- [GenericAgent](https://github.com/lsdefine/GenericAgent) — Self-evolving agent that grows its own skill tree from ~3K lines of seed code. 9 atomic tools for full system control (browser, terminal, filesystem, screen vision) with automatic skill crystallization.
- [Haystack](https://github.com/deepset-ai/haystack) — End-to-end NLP framework with agent pipelines.
- [Instructor](https://github.com/jxnl/instructor) — Structured output from LLMs. Essential for reliable tool use.
- [LangChain](https://github.com/langchain-ai/langchain) — The most popular LLM application framework. Agents, chains, tools.
- [LlamaIndex](https://github.com/run-llama/llama_index) — Data framework for LLM apps. Strong RAG and data agent support.
- [PydanticAI](https://github.com/pydantic/pydantic-ai) — GenAI agent framework, the Pydantic way. Type-safe and production-ready.
- [smolagents](https://github.com/huggingface/smolagents) — Hugging Face's lightweight agent library. ~1,000 lines of focused code, easy to understand and extend.

### Code Agents
- [Aider](https://github.com/paul-gauthier/aider) — AI pair programming in the terminal.
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — Anthropic's agentic coding tool. Terminal-based, strong at complex refactors and multi-file changes.
- [Codex](https://openai.com/index/introducing-codex/) — OpenAI's cloud-based coding agent. Runs tasks in sandboxed environments, integrates with GitHub.
- [Cursor](https://cursor.sh/) — AI-first code editor with agent capabilities.
- [Devin](https://devin.ai/) — Cognition's autonomous software engineer. Full environment with browser, editor, and terminal.
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) — Open-source AI agent bringing Gemini directly into your terminal.
- [GitHub Copilot](https://github.com/features/copilot) — AI pair programmer with agent mode for multi-file edits, terminal commands, and autonomous task execution.
- [Kiro](https://kiro.dev/) — AWS's spec-driven AI coding IDE. Three-phase Specify, Plan, Execute workflow.
- [Open SWE](https://github.com/langchain-ai/open-swe) — LangChain's open-source async cloud coding agent. Connects to GitHub repos, delegates tasks from issues via Slack or Linear.
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) — AI software development agent (formerly OpenDevin).
- [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk) — Modular Python SDK for building code agents. Local or ephemeral workspaces, composable tools, powers OpenHands CLI and Cloud.
- [SWE-agent](https://github.com/princeton-nlp/SWE-agent) — Princeton's software engineering agent.
- [Windsurf](https://windsurf.com/) — AI-native IDE by Codeium with agentic Cascade flows.

### Personal AI Agents
- [CoPaw](https://github.com/agentscope-ai/CoPaw) — Alibaba's open-source personal AI agent workstation. Supports multi-channel workflows, MCP skills, local/cloud LLMs, and persistent memory.
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — Nous Research's open-source self-improving personal AI agent. Closed learning loop, multi-platform gateway (Telegram, Discord, Slack, WhatsApp, Signal), MCP integration, and cron scheduling.
- [OpenClaw](https://github.com/openclaw/openclaw) — Open-source personal AI agent with tool use, browser control, messaging integration, and persistent memory.

### Browser Agents
- [Browser Use](https://github.com/browser-use/browser-use) — Control browsers with AI agents. Most popular browser automation framework.
- [Playwright MCP](https://github.com/anthropics/anthropic-tools) — Anthropic's browser automation via MCP.
- [Stagehand](https://github.com/browserbase/stagehand) — AI-powered browser automation framework by Browserbase.
- [UI-TARS Desktop](https://github.com/bytedance/UI-TARS-desktop) — ByteDance's multimodal AI agent stack for desktop automation.

### Research Agents
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) — Autonomous agent for deep research on any topic using any LLM.
- [autoresearch](https://github.com/karpathy/autoresearch) — Andrej Karpathy's open-source framework for running AI agents that autonomously conduct research on single-GPU model training experiments overnight.
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) — Open-source AI-powered answering engine (Perplexity alternative).

## Platforms & Low-Code

- [Activepieces](https://github.com/activepieces/activepieces) — Open-source AI workflow automation with 400+ MCP servers for agents.
- [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/) — AWS managed agent service.
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) — All-in-one desktop & Docker AI app with built-in RAG, agents, and MCP.
- [Anthropic Claude + Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Claude's function calling and agent capabilities.
- [Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/overview) — Anthropic's hosted agent execution environment (public beta, April 2026). Stateful sessions, built-in sandboxing, and tool execution without managing your own infrastructure.
- [Azure AI Foundry](https://ai.azure.com/) — Full-stack AI platform with agent capabilities.
- [Composio](https://github.com/ComposioHQ/composio) — 1000+ toolkits, auth management, and sandboxed workbench for AI agents.
- [Dify](https://github.com/langgenius/dify) — Open-source LLMOps platform with visual agent builder.
- [Google Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/agents) — Google Cloud's agent development platform.
- [MaxKB](https://github.com/1Panel-dev/MaxKB) — Open-source platform for building enterprise-grade agents.
- [Microsoft Copilot Studio](https://copilotstudio.microsoft.com/) — Low-code agent builder. Integrates with M365, Dynamics, Power Platform.
- [n8n](https://n8n.io/) — Workflow automation with native AI agent capabilities and MCP support.
- [OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview) — OpenAI's managed agent platform with tools and retrieval.
- [Relevance AI](https://relevanceai.com/) — No-code AI agent platform.
- [Trigger.dev](https://github.com/triggerdotdev/trigger.dev) — Build and deploy fully-managed AI agents and workflows.

## Agent Infrastructure

### Tool Protocols
- [Agent2Agent Protocol (A2A)](https://github.com/google/A2A) — Google's open protocol for agent-to-agent communication and discovery. Linux Foundation project.
- [Context7](https://github.com/upstash/context7) — MCP server for up-to-date code documentation for LLMs.
- [GitHub MCP Server](https://github.com/github/github-mcp-server) — GitHub's official MCP server for AI agents.
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) — Anthropic's standard for connecting AI to tools and data.
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — De facto standard for LLM tool use.


### Agent Skills & Tools
- [PowerSkills](https://github.com/aloth/PowerSkills) — PowerShell automation toolkit for AI agents. Structured JSON control over Windows — Outlook, Edge browser, desktop, and system operations.

### Memory & State
- [Hindsight](https://github.com/vectorize-io/hindsight) — Agent memory that learns: state-of-the-art memory layer for AI agents with persistent, personalized recall.
- [Letta](https://github.com/letta-ai/letta) — Stateful agents with long-term memory (formerly MemGPT).
- [Mem0](https://github.com/mem0ai/mem0) — Universal memory layer for AI agents. Persistent, contextual.
- [Zep](https://github.com/getzep/zep) — Long-term memory for AI assistants.

### Monitoring & Observability
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — ML & LLM observability.
- [Helicone](https://www.helicone.ai/) — LLM observability and cost tracking.
- [Langfuse](https://github.com/langfuse/langfuse) — Open-source LLM observability. Traces, evals, prompt management.
- [LangSmith](https://smith.langchain.com/) — LangChain's debugging and monitoring platform.

### Data Extraction
- [Crawl4AI](https://github.com/unclecode/crawl4ai) — Open-source LLM-friendly web crawler. High-performance async crawling.
- [Firecrawl](https://github.com/firecrawl/firecrawl) — Turn entire websites into LLM-ready markdown or structured data.

### Vector Databases
- [Azure AI Search](https://azure.microsoft.com/products/ai-services/ai-search) — Enterprise search with vector + hybrid capabilities.
- [ChromaDB](https://github.com/chroma-core/chroma) — Lightweight embedding database.
- [Pinecone](https://www.pinecone.io/) — Managed vector database.
- [Qdrant](https://github.com/qdrant/qdrant) — High-performance vector search.
- [Weaviate](https://github.com/weaviate/weaviate) — Open-source vector database.

### Sandboxing & Execution
- [Daytona](https://github.com/daytonaio/daytona) — Secure and elastic infrastructure for running AI-generated code.
- [E2B](https://github.com/e2b-dev/e2b) — Cloud sandboxes for AI agents. Secure code execution environments.
- [GitHub Agentic Workflows](https://github.blog/changelog/2026-02-13-github-agentic-workflows-are-now-in-technical-preview/) — AI agents running within GitHub Actions. Markdown-based workflow definitions.
- [Moltworker](https://github.com/cloudflare/moltworker) — Cloudflare's open-source framework for deploying personal AI agents on Workers with sandboxed execution.

## Evaluation & Testing

- [AgentBench](https://github.com/THUDM/AgentBench) — Tsinghua's multi-dimensional agent benchmark.
- [AgentBoard](https://github.com/hkust-nlp/AgentBoard) — Multi-round agent evaluation platform.
- [GAIA](https://huggingface.co/gaia-benchmark) — General AI Assistants benchmark by Meta.
- [LangTest](https://github.com/Pacific-AI-Corp/langtest) — Testing framework for delivering safe & effective language models.
- [RuLES](https://github.com/normster/llm_rules) — Benchmark for evaluating rule-following in language models.
- [SWE-bench](https://www.swebench.com/) — Benchmark for software engineering agents.
- [ToolBench](https://github.com/OpenBMB/ToolBench) — Benchmark for tool-use capabilities.
- [ToolEmu](https://github.com/ryoungj/ToolEmu) — LM-based emulation framework for identifying risks of agents with tool use (ICLR '24).
- [UQLM](https://github.com/cvs-health/uqlm) — Uncertainty quantification for LLMs. UQ-based hallucination detection.

## Safety & Governance

- [Agent Governance Toolkit](https://github.com/microsoft/agent-governance-toolkit) — Microsoft's runtime governance infrastructure for AI agents. Deterministic policy enforcement, zero-trust identity, execution sandboxing, and SRE. Covers all 10 OWASP Agentic Top 10 risks across Python, TypeScript, .NET, Rust, and Go.
- [Agentic Security](https://github.com/msoedov/agentic_security) — LLM vulnerability scanner and AI red teaming kit.
- [Anthropic Constitutional AI](https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback) — Self-improving AI safety through constitutions.
- [Azure AI Content Safety](https://azure.microsoft.com/products/ai-services/ai-content-safety) — Content moderation for AI outputs.
- [Guardrails AI](https://github.com/guardrails-ai/guardrails) — Validation framework for LLM outputs.
- [IronCurtain](https://github.com/provos/ironcurtain) — Open-source security layer for autonomous AI agents. Runs agents in isolated VMs to prevent prompt injection and rogue behavior.
- [LangFair](https://github.com/cvs-health/langfair) — Python library for LLM bias and fairness assessments.
- [LLM Guard](https://github.com/protectai/llm-guard) — Security toolkit for LLM interactions.
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) — NVIDIA's programmable guardrails.
- [PromptInject](https://github.com/agencyenterprise/PromptInject) — Framework for quantitative analysis of LLM robustness to prompt attacks (NeurIPS '22 Best Paper).
- [Rebuff](https://github.com/protectai/rebuff) — Prompt injection detection.
- [Safe RLHF](https://github.com/PKU-Alignment/safe-rlhf) — Constrained value alignment via safe reinforcement learning from human feedback.

## Research Papers

### Surveys & Overviews
- [The Rise and Potential of Large Language Model Based Agents](https://arxiv.org/abs/2309.07864) (2023) — Comprehensive survey of LLM-based agents.
- [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432) (2023) — Systematic review of agent architectures.
- [Agent AI: Surveying the Horizons of Multimodal Interaction](https://arxiv.org/abs/2401.03568) (2024) — Microsoft Research survey on agent AI.

### Agent Architectures
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) (2023) — The foundational Reason + Act paradigm.
- [Toolformer](https://arxiv.org/abs/2302.04761) (2023) — Teaching LLMs to use tools autonomously.
- [Voyager](https://arxiv.org/abs/2305.16291) (2023) — Lifelong learning agent in Minecraft.
- [Generative Agents](https://arxiv.org/abs/2304.03442) (2023) — Stanford's believable simulacra of human behavior.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) (2023) — Deliberate problem solving through exploration of reasoning paths.
- [Self-Refine](https://arxiv.org/abs/2303.17651) (2023) — Iterative self-refinement with self-feedback.

### Multi-Agent Systems
- [CAMEL](https://arxiv.org/abs/2303.17760) (2023) — Communicative agents for role-playing.
- [MetaGPT](https://arxiv.org/abs/2308.00352) (2023) — Multi-agent collaboration mimicking software companies.
- [ChatDev](https://arxiv.org/abs/2307.07924) (2023) — Agents collaborating in a virtual software company.
- [PaperOrchestra](https://arxiv.org/abs/2604.05018) (2026) — Google's multi-agent framework for automated AI research paper writing, converting unstructured pre-writing materials into submission-ready papers.

### Safety & Evaluation
- [AgentBench](https://arxiv.org/abs/2308.03688) (2023) — Evaluating LLMs as agents across 8 environments.
- [InjectAgent](https://arxiv.org/abs/2403.02691) (2024) — Indirect prompt injection attacks on tool-integrated agents.
- [R-Judge](https://arxiv.org/abs/2401.10019) (2024) — Benchmarking safety risk awareness for LLM agents.

### Agent Training
- [Group-in-Group Policy Optimization for LLM Agent Training](https://github.com/langfengQ/verl-agent) (2025) — RL-based training for LLM/VLM agents.

## Tutorials & Courses

- [DeepLearning.AI: A2A Protocol](https://www.deeplearning.ai/short-courses/a2a-the-agent2agent-protocol/) — Short course on Google's Agent2Agent protocol.
- [DeepLearning.AI: Building Agentic RAG](https://www.deeplearning.ai/) — Andrew Ng's course on agentic RAG patterns.
- [Hugging Face: Building AI Agents](https://huggingface.co/learn/agents-course/) — Open course on agent development.
- [LangChain Academy](https://academy.langchain.com/) — Free courses on agents and RAG.
- [Microsoft: AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) — 12 lessons to get started building AI agents.
- [Microsoft: Build AI Agents with Azure AI Foundry](https://learn.microsoft.com/training/) — Official Microsoft Learn path.
- [Microsoft: MCP for Beginners](https://github.com/microsoft/mcp-for-beginners) — Curriculum for Model Context Protocol with cross-language examples.
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) — Comprehensive guides for prompt engineering, RAG, and AI agents.

## Use Cases & Case Studies

### Enterprise
- IT Helpdesk Agents — Automated ticket resolution, password resets
- Customer Service — Multi-turn conversation with CRM integration
- Document Intelligence — Contract analysis, compliance checking
- Data Analysis — Natural language to SQL, automated reporting

### Research & Humanitarian
- Disinformation Detection — Agents monitoring information ecosystems
- Disaster Response — Coordinating information flows in crisis situations
- Knowledge Management — Intelligent document retrieval for NGOs

## Community

- [r/AI_Agents](https://www.reddit.com/r/AI_Agents/) — Reddit community
- [AI Agents Discord](https://discord.gg/ai-agents) — Active Discord server
- [awesome-ai-agent-papers](https://github.com/VoltAgent/awesome-ai-agent-papers) — Curated collection of AI agent research papers released in 2026, covering engineering, memory, evaluation, workflows, and autonomous systems.
- [#AIAgents on X](https://x.com/search?q=%23AIAgents) — Twitter/X hashtag

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

---

> *Disclaimer: This list aims to be vendor-neutral and community-driven. Inclusion does not imply endorsement by any employer or organization.*
