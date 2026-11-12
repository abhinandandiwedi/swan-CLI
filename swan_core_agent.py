import os
import json
import subprocess
import requests
from pathlib import Path

class SwanCoreAgent:
    def __init__(self, model_name="swan-coder", workspace_path="~/-TIDDA-Core"):
        self.model = model_name
        self.endpoint = "http://localhost:11434/api/generate"
        self.workspace = Path(workspace_path).expanduser().resolve()
        self.repo_map_path = self.workspace / "docs" / "repo_map.md"

    def _get_repo_map_preview(self) -> str:
        if self.repo_map_path.exists():
            return self.repo_map_path.read_text(encoding="utf-8")[:6000]
        return "Repo map not generated yet."

    def _build_system_prompt(self) -> str:
        return f"""You are SWAN-ENGINE, a surgical autonomous software engineering agent working on TIDDA-Core.
Workspace Root: {self.workspace}

Here is the indexed AST Map of the repository so you already know where every file, class, and function lives:
{self._get_repo_map_preview()}

CRITICAL RULES:
1. Do NOT wander around reading random files if the answer is already in the AST Map or previous observations.
2. NEVER repeat the exact same tool call with the exact same parameters.
3. As soon as you have enough information to answer or complete the User Objective, IMMEDIATELY call the "finish" tool with a detailed answer in "message".

Respond ONLY with a raw JSON object matching this schema:
{{
  "thought": "Brief reasoning about what is needed next",
  "tool": "write_file" | "read_file" | "run_command" | "list_dir" | "finish",
  "params": {{
      // write_file: "path", "content"
      // read_file: "path"
      // run_command: "command"
      // list_dir: "path"
      // finish: "message"
  }}
}}"""

    def _call_model(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": 16384}
        }
        res = requests.post(self.endpoint, json=payload, timeout=180)
        return json.loads(res.json().get("response", "{}"))

    def execute_task(self, objective: str, max_steps: int = 8):
        print(f"\n[MISSION] {objective}")
        context = f"{self._build_system_prompt()}\n\nUser Objective: {objective}\n"
        seen_actions = set()

        for step in range(1, max_steps + 1):
            print(f"\n--- Step {step}/{max_steps} ---")
            action = self._call_model(context)
            thought = action.get("thought", "")
            tool = action.get("tool", "")
            params = action.get("params", {})

            print(f"🧠 Thought: {thought}")
            print(f"🛠 Tool: {tool} | Params: { {k: (v[:40] + '...' if isinstance(v, str) and len(v) > 40 else v) for k, v in params.items()} }")

            if tool == "finish":
                print(f"\n✔ MISSION COMPLETE:\n{params.get('message')}")
                return

            action_signature = f"{tool}:{json.dumps(params, sort_keys=True)}"
            if action_signature in seen_actions:
                result = "WARNING: You already executed this exact tool and parameters! Do NOT repeat it. Either move to the next logical step or call 'finish' immediately."
                print(f"⚠ Loop prevented!")
                context += f"\nStep {step} Action: {json.dumps(action)}\nObservation: {result}\n"
                continue
            seen_actions.add(action_signature)

            result = ""
            try:
                if tool == "list_dir":
                    target = self.workspace / params.get("path", ".")
                    files = [p.name for p in target.iterdir() if not p.name.startswith('.')]
                    result = "\n".join(files)
                elif tool == "read_file":
                    target = self.workspace / params.get("path", "")
                    result = target.read_text(encoding="utf-8")[:4000]
                elif tool == "write_file":
                    target = self.workspace / params.get("path", "")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(params.get("content", ""), encoding="utf-8")
                    result = f"Successfully wrote to {target}"
                elif tool == "run_command":
                    proc = subprocess.run(params.get("command", ""), shell=True, cwd=self.workspace, capture_output=True, text=True, timeout=30)
                    result = f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nCODE:{proc.returncode}"
                else:
                    result = f"Unknown tool '{tool}'. Use finish if done."
            except Exception as e:
                result = f"Error: {e}"

            print(f"📄 Result preview: {result[:250].replace(chr(10), ' ')}...")
            context += f"\nStep {step} Action: {json.dumps(action)}\nObservation: {result}\n"

        print("\n⚠ Max steps reached without explicit finish call.")

if __name__ == "__main__":
    agent = SwanCoreAgent()
    print("⚡ SWAN-ENGINE v2 Online (16k Context + AST Repo Map Loaded)")
    try:
        while True:
            cmd = input("\nSWAN-CMD >> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["exit", "quit"]:
                break
            agent.execute_task(cmd)
    except KeyboardInterrupt:
        print("\nShutting down SWAN-ENGINE.")
