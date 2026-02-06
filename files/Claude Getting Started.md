# Getting Started with Claude Code

Claude Code (CC) is a terminal-based agentic coding assistant from Anthropic that has become a staple for developers due to its ability to read, edit, and execute code directly. However, for power users, its official pricing is a significant barrier, often costing between $100 and $200 per month for high-volume usage. This has created a demand for more affordable alternatives that still offer the robust automation and workflow benefits of the CC interface.

The solution recently arrived via Ollama, which now supports the Anthropic Messages API. By redirecting CC's API calls to a local Ollama instance, users can run the tool for free using open-source models like Qwen3-Coder or GPT-OSS. While there is a slight "capability hit" compared to Anthropic’s hosted Claude 3.5 or 4.0 models—largely because local models may struggle with CC's massive 16.5K-token system prompt—it provides a cost-effective, private, and offline-capable alternative for daily development tasks.

## Installation

* In PowerShell irm https://claude.ai/install.ps1 | iex

## Ollama

* ollama pull gpt-oss:20b for a local model
* claude --model gpt-oss:120b-cloud for a free-ish much larger cloud model
* See the Ollama Chat App for more models

## Configure 

* PS C:\Users\XXX> $env:ANTHROPIC_AUTH_TOKEN = "ollama"
* PS C:\Users\XXX> $env:ANTHROPIC_API_KEY = ""
* PS C:\Users\XXX> $env:ANTHROPIC_BASE_URL = "http://localhost:11434"
* PS C:\Users\XXX> claude --model gpt-oss:20b
* Ctr d to quit

## Teach Me Claude Code

### The "Agentic" Workflow
* To get the best results, don't just ask Claude to "write code." Follow the Research → Plan → Execute loop:
* Explore: Ask, "What does this project do?" or "Where is the logic for user auth?"
* Plan: Ask, "Create a plan to add a dark mode toggle to the settings page."
* Review: Claude will show you a plan. You can say, "Looks good, but use Tailwind classes instead of CSS."
* Execute: Say, "Go ahead and implement that." Claude will write the files and, if you allow it, run tests to verify the fix.

#### Pro Tip: The CLAUDE.md File
* As we've discussed regarding originality, the way you manage context is everything. CLAUDE.md is your project’s "memory." Claude reads this at the start of every session.
* Include these for a high-quality experience:
* Tech Stack: Specific versions (e.g., Node 20, Next.js 14).
* Coding Standards: "We prefer functional components over classes."
* Commands: How to run your specific test suite or build command.

#### Originality Check
* If you are building a project around Claude Code, the "standard" use case is just using it to write code faster.
* How to make your Claude Code project more original:
    * MCP Integration: Use the Model Context Protocol (MCP) to give Claude access to your Jira tickets, Google Drive docs, or Slack. Instead of just coding, Claude can now "Update the Jira ticket \#402 with the results of the refactor I just did."
    * TDD Agent: Configure CC to strictly follow Test-Driven Development. Tell it: "You are not allowed to commit code until you have written a failing test and then made it pass."