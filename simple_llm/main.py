import click
from query import query_bot
import dotenv

dotenv.load_dotenv()

@click.group()
def cli():
    pass

@cli.command()
@click.argument("question")
@click.option("--show-sources", is_flag=True, help="Show retrieved sources")
def query(question, show_sources):
    """Ask the chatbot a question."""
    answer = query_bot(question, show_sources=show_sources)
    click.echo("\n🤖 Answer:\n")
    click.echo(answer)

if __name__ == "__main__":
    cli()