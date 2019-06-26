import json

from wumpus.neuralnetwork import train, create_network


def configure(config: str = "config.json"):
    with open(config, 'r') as c:
        data = json.loads('\n'.join(c.readlines()))
    inp = list()
    out = list()
    queries = list()
    for module in data["modules"]:
        for question in module["questions"]:
            inp.append(question)
            out.append(data["modules"].index(module))
        queries.append(module["query_name"])
    create_network(len(data["modules"]))
    train(inp, out)
    return queries
