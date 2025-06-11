import logging
import os
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from app.factory.agent_factory import create_agent_team
from app.orchestration.orchestrator import Orchestrator

# Create a typer app
app = typer.Typer()

def setup_logging():
    """Configures the logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[
            logging.FileHandler("agent_activity.log", mode='w'),
            logging.StreamHandler()
        ]
    )

@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="The high-level goal for the agent team to accomplish.")],
    config: Annotated[str, typer.Option(help="Path to the main configuration file.")] = "configs/config.yaml"
):
    """
    Runs the multi-agent system to accomplish a given goal.
    """
    load_dotenv()
    setup_logging()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        typer.echo("Error: Please make sure your .env file contains OPENAI_API_KEY.")
        raise typer.Exit(code=1)

    try:
        # Create the agent team using the factory
        task_decomposer, worker_agents = create_agent_team(config)

        # Initialize and run the orchestrator
        orchestrator = Orchestrator(
            task_decomposer=task_decomposer,
            worker_agents=worker_agents
        )
        
        typer.echo(f"🚀 Starting agent team to accomplish goal: {goal}")
        orchestrator.run(goal)
        typer.echo("✅ Goal accomplished successfully!")

    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        typer.echo(f"Error: Configuration file not found at '{config}'. Please check the path.")
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled by user. Exiting...")
        logging.info("User cancelled the operation.")
        raise typer.Exit()
    except Exception as e:
        logging.critical("A critical error occurred: %s", e, exc_info=True)
        typer.echo(f"\n🚨 Critical Error: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app() 