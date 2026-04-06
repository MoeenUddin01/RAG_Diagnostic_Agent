## 🛸 Antigravity Task & Planning Logic

The Agent must operate in a **Stateless Task Cycle**. No code changes are permitted without an approved `task.md`.

### PHASE 1: THE BLUEPRINT (Automatic Creation)
When a new task is assigned, the Agent must immediately generate a `task.md` file. 
**The file must follow this exact structure:**
- **Objective:** Concise summary of the end goal.
- **Architectural Logic:** A 1-2 sentence explanation of the technical "why" behind the approach.
- **The Execution Flow:** A numbered list of steps the Agent will take.
- **File Impact Matrix:**
    - `[+] NEW`: List of files to be created.
    - `[*] MOD`: List of files to be edited.
- **Gatekeeper:** A bold footer saying: **"Waiting for 'PROCEED' to execute."**

### PHASE 2: THE APPROVAL GATE
The Agent must **STOP** all execution after creating `task.md`. It cannot modify, create, or delete any other files until the user explicitly sends the command **"PROCEED"**.

### PHASE 3: EXECUTION & AUTO-PURGE
Upon receiving "PROCEED":
1. Execute the changes in the order listed in the Flow.
2. Verify the task is functional (run linter or basic check).
3. **CRITICAL:** Once the task is complete, the Agent must **DELETE** the `task.md` file immediately.
4. Final response must be: *"Task complete. Short-lived context (task.md) has been purged."*