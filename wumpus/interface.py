from wumpus.dataloader import configure
from wumpus.actionnetwork import answer
from wumpus.argumentnetwork import get_arg_index
import wumpus.queries

queries, arguments, premades = configure()


def compute(question: str):
    index = answer(question)
    if arguments[index] is True:
        arg_index = get_arg_index(question, premades[queries[index]][0][:3])
    else:
        arg_index = -1
    name = queries[index]
    method = getattr(wumpus.queries, name)
    if arg_index >= 0:
        return method(question[arg_index:])
    else:
        return method()


if __name__ == "__main__":
    print(compute("Who are you?"))
    print(compute("Show me the status of the Discord servers"))
    print(compute("Tell me what you can do"))
