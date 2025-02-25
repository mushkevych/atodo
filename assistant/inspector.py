class ToolInvocationInspector:
    """ Inspect the tool calls for Trustcall """
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == 'chat_model':
                self.called_tools.append(
                    r.outputs['generations'][0][0]['message']['kwargs']['tool_calls']
                )


def extract_tool_info(tool_calls, schema_name='Memory'):
    """Extract information from tool calls for:
     - patches
     - new memories in Trustcall.
    
    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "UserProfile")
    """
    # Initialize list of changes
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                # Check if there are any patches
                if call['args']['patches']:
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                else:
                    # Handle case where no changes were needed
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f'Document {change['doc_id']} updated:\n'
                f'Plan: {change['planned_edits']}\n'
                f'Added content: {change['value']}'
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f'Document {change['doc_id']} unchanged:\n'
                f'{change['planned_edits']}'
            )
        else:
            result_parts.append(
                f'New {schema_name} created:\n'
                f'Content: {change['value']}'
            )
    
    return '\n\n'.join(result_parts)
