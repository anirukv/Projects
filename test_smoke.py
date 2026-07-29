from graph import build_graph, create_initial_state, run_graph

print('Building graph...')
graph = build_graph()
print('Graph built.')
state = create_initial_state(user_email='smoke@test', allowed_companies=['Google'], messages=[])
state['current_question'] = 'What did Google say about revenue in the latest call?'
print('Invoking graph...')
result = run_graph(graph, 'thread::smoke', state)
print('Answer:', result.get('answer', '')[:100])
print('Messages length:', len(result.get('messages', [])))
