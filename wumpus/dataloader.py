import json

from wumpus import actionnetwork
from wumpus import argumentnetwork


def configure(config: str = "config.json"):
    with open(config, 'r') as c:
        data = json.loads('\n'.join(c.readlines()))

    # input for action network
    inp = list()
    # output for action network
    out = list()
    # list of all available queries
    queries = list()
    # wether an argument is need - same order as queries
    arguments = list()
    # all questions with their queries
    premades = dict()
    # input and output for argument network
    query = dict()

    for q in data["data"]:
        if not premades.get(q["output"]): premades[q["output"]] = [q["input"]]
        else: premades[q["output"]].append(q["input"])

        if not q["output"] in queries:
            queries.append(q["output"])
            if q["args"] != -1: arguments.append(True)
            else: arguments.append(False)

        if query.get(q["output"]) and len(query.get(q["output"])) < 4:
            query.get(q["output"])[0].append(q["input"])
        elif not query.get(q["output"]):
            if q["args"] >= 0:
                query[q["output"]] = [[q["input"]], [q["args"]]]

        inp.append(q["input"])
        out.append(queries.index(q["output"]))

    actionnetwork.create_network(len(queries))
    actionnetwork.train(inp, out)

    arg_in = [query.get(e, [[], []])[0][:4] for e in queries]
    arg_out = [query.get(e, [[], []])[1][:4] for e in queries]
    entries = 0
    for i in arg_in: entries += len(i)
    for o in arg_out: entries += len(o)

    if entries > 0:
        argumentnetwork.create_network()
        argumentnetwork.train(arg_in, arg_out)

    return queries, arguments, premades
