import openai
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.logging import RichHandler
from rich.columns import Columns
from rich import print
import json
import time
import random
from pathlib import Path
import logging
from datetime import datetime

# Set up console and logging
console = Console()

# Configure logging to use rich handler but simpler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_time=True, show_path=False)]
)
logger = logging.getLogger(__name__)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
MAX_QUESTIONS = 1000  # Prevent infinite generation

class MetricsTracker:
    def __init__(self):
        self.start_time = datetime.now()
        self.questions_generated = 0
        self.questions_failed = 0
        self.api_calls = 0
        self.retries = 0
        self.buffer_size = 0
        
    def get_runtime(self):
        return datetime.now() - self.start_time
        
    def get_rate(self):
        runtime_seconds = self.get_runtime().total_seconds()
        if runtime_seconds > 0:
            return self.questions_generated / runtime_seconds * 60  # questions per minute
        return 0
        
    def display_stats(self):
        """Display current statistics in a nice table."""
        runtime = self.get_runtime()
        rate = self.get_rate()
        
        table = Table(title="📊 Generation Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="bright_green", width=15)
        
        table.add_row("⏱️ Runtime", f"{str(runtime).split('.')[0]}")
        table.add_row("✅ Generated", f"{self.questions_generated}")
        table.add_row("❌ Failed", f"{self.questions_failed}")
        table.add_row("🔄 API Calls", f"{self.api_calls}")
        table.add_row("🔁 Retries", f"{self.retries}")
        table.add_row("📦 Buffer Size", f"{self.buffer_size}")
        table.add_row("⚡ Rate", f"{rate:.1f}/min")
        
        console.print(table)

metrics = MetricsTracker()

def display_header():
    """Display a nice header."""
    console.print(Panel.fit(
        "[bold blue]🤖 Dataset Evolution Generator[/bold blue]\n"
        "[dim]Generating creative questions from seed dataset...[/dim]",
        border_style="blue",
        title="Welcome"
    ))

def display_question_result(count, question, answer):
    """Display a generated question in a nice format."""
    # Truncate long answers for display
    display_answer = answer[:150] + "..." if len(answer) > 150 else answer
    
    console.print(Panel(
        f"[bold cyan]Q{count}:[/bold cyan] {question}\n\n"
        f"[bold yellow]A:[/bold yellow] {display_answer}",
        title=f"✅ Question {count} Generated",
        border_style="green",
        width=80
    ))

def llm_response(prompt: str, num_samples: int = 1, retries: int = MAX_RETRIES):
    """Call OpenAI API with retry logic and error handling."""
    metrics.api_calls += 1
    
    for attempt in range(retries + 1):
        try:
            resp = client.responses.create(
                model="o4-mini",
                tools=[
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto"}
                    }
                ],
                input=prompt,
            )
            return resp.output
        except Exception as e:
            metrics.retries += 1
            logger.warning(f"API call failed (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                # Exponential backoff with jitter
                delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                logger.error(f"API call failed after {retries + 1} attempts")
                raise

def generate_new_question(question: str):
    """Generate a new question with error handling."""
    try:
        if not question or not question.strip():
            logger.warning("Empty question provided, skipping")
            return None
            
        prompt = (
            f"Given the following question: {question}\n"
            "generate a new question to be creative are different from the original question"
            "give only the question and nothing else"
        )
        response = llm_response(prompt)
        
        if not response or len(response) == 0:
            logger.warning("Empty response from API")
            return None
            
        new_question = response[-1].content[-1].text
        
        if not new_question or not new_question.strip():
            logger.warning("Generated question is empty")
            return None
            
        return new_question.strip()
        
    except Exception as e:
        logger.error(f"Error generating new question: {e}")
        return None

def using_tool_call(response) -> bool:
    """Check if response uses tool calls."""
    try:
        return any("response_code_interpreter_tool_call" in str(type(item)) for item in response)
    except Exception as e:
        logger.warning(f"Error checking tool call: {e}")
        return False

def verify_question(question: str):
    """Verify question and get answer with error handling."""
    try:
        if not question or not question.strip():
            logger.warning("Empty question provided for verification")
            return None
            
        prompt = (
            f"{question}\n"
            "give answer to above question in <answer></answer> tags"
            "give only the answer and nothing else"
        )
        
        response = llm_response(prompt)
        
        if not response or len(response) == 0:
            logger.warning("Empty response from verification API")
            return None
        
        if not using_tool_call(response):
            logger.warning("Response does not use tool call")
            return None
        
        text = response[-1].content[-1].text.strip()
        
        if "<answer>" in text and "</answer>" in text:
            answer = text.split("<answer>")[1].split("</answer>")[0].strip()
            if answer:
                return answer
        
        logger.warning("No valid answer found in response")
        return None
        
    except Exception as e:
        logger.error(f"Error verifying question: {e}")
        return None

def load_seed_dataset(filepath: str):
    """Load seed dataset with error handling."""
    try:
        if not os.path.exists(filepath):
            logger.error(f"Seed dataset file not found: {filepath}")
            return None
            
        with open(filepath, "r", encoding='utf-8') as f:
            dataset = json.load(f)
            
        if not isinstance(dataset, list):
            logger.error("Dataset should be a list")
            return None
            
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return None
            
        # Validate dataset structure
        valid_questions = []
        for i, item in enumerate(dataset):
            if isinstance(item, dict) and "problem" in item:
                if item["problem"] and item["problem"].strip():
                    valid_questions.append(item["problem"].strip())
                else:
                    logger.warning(f"Empty problem at index {i}")
            else:
                logger.warning(f"Invalid item structure at index {i}")
                
        if not valid_questions:
            logger.error("No valid questions found in dataset")
            return None
            
        logger.info(f"✅ Loaded {len(valid_questions)} valid questions from {len(dataset)} total items")
        return valid_questions
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def ensure_output_directory(filepath: str):
    """Ensure output directory exists."""
    try:
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        raise

def main():
    """Main function with comprehensive error handling and rich UI."""
    try:
        # Display header
        display_header()
        console.print()
        
        # Load seed dataset
        logger.info("📂 Loading seed dataset...")
        questions = load_seed_dataset("seed_dataset.json")
        if not questions:
            logger.error("❌ Failed to load seed dataset")
            return
            
        buffer = questions.copy()
        count = 0
        failed_count = 0
        max_questions = MAX_QUESTIONS
        
        # Ensure output directory exists
        output_file = "new_dataset.jsonl"
        ensure_output_directory(output_file)
        
        logger.info(f"🚀 Starting generation with {len(buffer)} seed questions")
        logger.info(f"🎯 Target: {max_questions} questions")
        logger.info(f"📁 Output file: {output_file}")
        console.print()
        
        # Setup progress tracking with a working progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False
        ) as progress:
            
            task = progress.add_task("🔄 Generating questions...", total=max_questions)
            
            with open(output_file, "a", encoding='utf-8') as f:
                while buffer and count < max_questions:
                    try:
                        if len(buffer) == 0:
                            logger.warning("⚠️ Buffer is empty, stopping")
                            break
                            
                        question = buffer.pop(0)
                        
                        # Update progress
                        progress.update(task, description=f"🔄 Processing question {count + 1}/{max_questions}...")
                        
                        # Update metrics
                        metrics.questions_generated = count
                        metrics.questions_failed = failed_count
                        metrics.buffer_size = len(buffer)
                        
                        # Retry loop for generating a question with valid answer
                        max_question_retries = MAX_RETRIES
                        new_question = None
                        answer = None
                        
                        for retry_attempt in range(max_question_retries + 1):
                            # Generate new question
                            new_question = generate_new_question(question)
                            if not new_question:
                                if retry_attempt < max_question_retries:
                                    delay = RETRY_DELAY * (2 ** retry_attempt) + random.uniform(0, 1)
                                    time.sleep(delay)
                                    continue
                                break
                                
                            # Verify the question
                            answer = verify_question(new_question)
                            if answer:
                                break
                            else:
                                if retry_attempt < max_question_retries:
                                    delay = RETRY_DELAY * (2 ** retry_attempt) + random.uniform(0, 1)
                                    time.sleep(delay)
                        
                        # Skip if we couldn't generate a valid question-answer pair
                        if not new_question or not answer:
                            failed_count += 1
                            logger.warning("⚠️ Failed to generate valid question-answer pair")
                            continue
                            
                        # Save to file
                        try:
                            record = {"problem": new_question, "answer": answer}
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f.flush()  # Ensure data is written
                            
                            count += 1
                            buffer.append(new_question)  # Add to buffer for further evolution
                            
                            # Update progress
                            progress.update(task, advance=1)
                            
                            # Display the generated question
                            display_question_result(count, new_question, answer)
                            
                            # Show stats every 5 questions
                            if count % 5 == 0:
                                console.print()
                                metrics.display_stats()
                                console.print()
                            
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"❌ Error writing to file: {e}")
                            continue
                            
                    except KeyboardInterrupt:
                        logger.info("⚠️ Generation interrupted by user")
                        break
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"❌ Error in main loop: {e}")
                        continue
        
        # Final summary
        console.print("\n" + "="*60)
        logger.info(f"🏁 Generation completed! Generated: {count}, Failed: {failed_count}")
        metrics.display_stats()
        
        console.print("\n")
        console.print(Panel.fit(
            f"[bold green]🎉 Generation Completed![/bold green]\n"
            f"[green]✅ Questions generated: {count}[/green]\n"
            f"[red]❌ Failed attempts: {failed_count}[/red]\n"
            f"[blue]📁 Output saved to: {output_file}[/blue]",
            border_style="green",
            title="Final Results"
        ))
        
    except KeyboardInterrupt:
        logger.info("⚠️ Program interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]❌ OPENAI_API_KEY environment variable is not set[/red]")
        exit(1)
        
    main()