import inspect
from graph import build_graph
from langchain_core.messages import HumanMessage, SystemMessage

g = build_graph()
closure = inspect.getclosurevars(g.get_graph().nodes['generate_node'].data.func).nonlocals
llm = closure.get('llm')
print('llmType', type(llm))
try:
    response = llm.invoke([
        SystemMessage(content='You answer strictly from the provided context.'),
        HumanMessage(content='Hello?'),
    ])
    print('response_type', type(response))
    print('content', getattr(response, 'content', None))
except Exception as exc:
    import traceback
    traceback.print_exc()
