from wumpus.dataloader import configure
from wumpus.neuralnetwork import answer
import wumpus.queries

queries = configure()


def compute(question: str):
    index = answer(question)
    name = queries[index]
    method = getattr(wumpus.queries, name)
    return method()


if __name__ == "__main__":
    print(compute("Who are you?"))
    print(compute("Show me the status of the Discord servers"))
    print(compute("Tell me what you can do"))
