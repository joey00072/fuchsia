import multiprocessing
import io
from contextlib import redirect_stdout, redirect_stderr
from fastmcp import FastMCP
from IPython.core.interactiveshell import InteractiveShell
from types import MethodType
from contextlib import redirect_stdout, redirect_stderr
import io
import signal

def run_code(code_string: str) -> str:
    """
    Execute a snippet of Python code in an IPython shell and return
    the combined stdout/stderr as a string.
    """
    ip = InteractiveShell.instance()
    ip.colors = "NoColor"                      # disable syntax coloring
    ip.ast_node_interactivity = "last_expr"    # only show last expression
    # suppress the In[]/Out[] prompts
    ip.displayhook.write_output_prompt = MethodType(
        lambda self, *a, **k: None,
        ip.displayhook
    )

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        ip.run_cell(code_string, store_history=False)
    return buf.getvalue().strip()

def _worker(code: str, out_queue: multiprocessing.Queue):
    output = run_code(code)
    out_queue.put(output)

def run_code_with_timeout(code: str, timeout: int = 3) -> str:
    """Run arbitrary Python code in a subprocess, killing it if it runs over `timeout` seconds."""
    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return "Error: Execution timed out"
    return q.get() if not q.empty() else ""

# Create the MCP server
mcp = FastMCP("InProcCodeRunner")

@mcp.tool
def execute(code: str) -> dict[str, str]:
    """Run Python code (max 3s) and return its stdout+stderr."""
    try:
        output = {"stdout/stderr": run_code_with_timeout(code)}
    except Exception as e:
        output = {"stdout/stderr": f"Error: {e}"}
    return output

if __name__ == "__main__":
    # You can expose this over standard IO (the default) or HTTP.
    # For an HTTP endpoint:
    mcp.run(
        transport="http",       # use Streamable HTTP transport
        host="0.0.0.0",         # listen on all interfaces
        port=8111,              # port of your choice
        path="/mcp"             # optional custom path
    )
    # Or for default stdio-based transport, simply:
    # mcp.run()
