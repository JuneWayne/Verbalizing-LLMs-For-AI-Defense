"""Evaluate the base model and reference-response adapters with local benchmark tools"""

from importlib import import_module
import argparse
import hashlib
import json
import re
import sys
import time

from scripts.curation.evaluation.task_benchmark_common import (
    CONFIG,
    LocalModel,
    METHODS,
    OUTPUT,
    ROOT,
    TOOL_PARSER_VERSION,
    UPSTREAM,
    block_network,
    digest,
    freeze_manifest,
    import_file,
    parse_tool_calls,
    summarize,
    write_json,
)


def dojo_suite_names(benchmark):
    return ['workspace', 'travel', 'banking', 'slack'] if benchmark == 'AgentDojo' else ['shopping', 'github', 'dailylife']


def load_dojo(benchmark):
    sys.path.insert(0, str(UPSTREAM / benchmark / 'src'))

    # load the suite implementation only after selecting the benchmark source checkout
    get_suite = import_module('agentdojo.task_suite.load_suites').get_suite
    return {name: get_suite('v1.2.2' if benchmark == 'AgentDojo' else 'v1', name) for name in dojo_suite_names(benchmark)}


def prepare(benchmark):
    rows = []
    if benchmark in ['AgentDojo', 'AgentDyn']:
        suites = load_dojo(benchmark)
        FixedJailbreakAttack = import_module('agentdojo.attacks.base_attacks').FixedJailbreakAttack
        ImportantInstructionsAttack = import_module('agentdojo.attacks.important_instructions_attacks').ImportantInstructionsAttack
        for name, suite in suites.items():
            # fixed published attack text is identical for every adapter
            attack = FixedJailbreakAttack(ImportantInstructionsAttack._JB_STRING, suite, None)
            attack.user_name = 'Emma Johnson'
            attack.model_name = 'Qwen2.5'
            for task_id, task in suite.user_tasks.items():
                rows.append({'id': f'{name}/{task_id}/safe', 'condition': 'safe', 'suite': name, 'task': task_id, 'injection_task': None, 'injections': {}})
                for injection_id, injection in suite.injection_tasks.items():
                    values = attack.attack(task, injection)
                    rows.append({'id': f'{name}/{task_id}/{injection_id}', 'condition': 'injection', 'suite': name, 'task': task_id, 'injection_task': injection_id, 'injections': values})
        details = {'scope': 'All user tasks and all injection goals in the selected suites, plus clean runs', 'suites': list(suites), 'attack': 'Published important_instructions template with victim name Qwen2.5, frozen across adapters', 'tool_parser_version': TOOL_PARSER_VERSION, 'changes': ['Local Hugging Face backend with Qwen native tool formatting', 'Complete standalone tagged calls accepted after explanatory text; fenced, inline and blockquoted examples excluded; omitted closing tags accepted only with complete JSON', 'Greedy generation, 1024 tokens per turn, native 15 tool-loop iterations', 'No adaptive attack optimization or paid APIs'], 'checks': 'Unchanged upstream task and injection success functions', 'invalid': 'Errors and missing scores remain unscored; they never count as successful defense'}
    else:
        for setting in ['base', 'enhanced']:
            for kind in ['dh', 'ds']:
                file = UPSTREAM / benchmark / 'data' / f'test_cases_{kind}_{setting}.json'
                for index, item in enumerate(json.loads(file.read_text())):
                    rows.append({'id': f'{setting}/{kind}/{index}', 'condition': 'injection', 'setting': setting, 'kind': kind, 'item': item})
        details = {'scope': 'All released base and enhanced cases, reported separately', 'changes': ['Qwen native chat template around the original InjecAgent system and user prompts', 'Greedy generation with 1024 output tokens', 'Use published simulated tool responses when present; missing responses use the frozen base model with the original simulator prompt', 'Local simulated responses are shared across adapters and identified in each record'], 'checks': 'Original evaluate_output_prompted function; no real tools execute', 'utility': 'No benchmark task-completion checker is provided; do not infer task success from action validity', 'invalid': 'Invalid formatting and missing simulated responses remain unscored'}
    freeze_manifest(benchmark, rows, details)


def dojo_pipeline(benchmark, generator):
    # bind the runtime classes from the benchmark checkout selected before constructing this pipeline
    AgentPipeline = import_module('agentdojo.agent_pipeline').AgentPipeline
    BasePipelineElement = import_module('agentdojo.agent_pipeline').BasePipelineElement
    InitQuery = import_module('agentdojo.agent_pipeline').InitQuery
    SystemMessage = import_module('agentdojo.agent_pipeline').SystemMessage
    ToolsExecutionLoop = import_module('agentdojo.agent_pipeline').ToolsExecutionLoop
    ToolsExecutor = import_module('agentdojo.agent_pipeline').ToolsExecutor
    load_system_message = import_module('agentdojo.agent_pipeline.agent_pipeline').load_system_message
    FunctionCall = import_module('agentdojo.functions_runtime').FunctionCall
    get_text_content_as_str = import_module('agentdojo.types').get_text_content_as_str
    text_content_block_from_string = import_module('agentdojo.types').text_content_block_from_string

    class LocalLLM(BasePipelineElement):
        name = 'Qwen2.5'

        def query(self, query, runtime, env, messages, extra_args):
            chat = []
            for message in messages:
                content = get_text_content_as_str(message['content']) if message['content'] else ''
                row = {'role': message['role'], 'content': content}
                if message['role'] == 'assistant' and message['tool_calls']:
                    row['tool_calls'] = [{'type': 'function', 'function': {'name': call.function, 'arguments': dict(call.args)}} for call in message['tool_calls']]
                if message['role'] == 'tool':
                    row['content'] = message.get('error') or content
                chat.append(row)

            # provide the model with the simulated tool names and argument schemas
            tools = [{'type': 'function', 'function': {'name': tool.name, 'description': tool.description, 'parameters': tool.parameters.model_json_schema()}} for tool in runtime.functions.values()]
            result = generator.generate(chat, tools)

            # parse the generated request before any simulated tool can execute
            calls, invalid = parse_tool_calls(result['text'], finished=result['finished'])
            result['invalid_tool_format'] = invalid
            result['tool_parser_version'] = TOOL_PARSER_VERSION
            result['parsed_tool_calls'] = calls

            # parser failures remain evaluation errors, rather than successful defenses
            if invalid:
                raise ValueError('tool_call_format: ambiguous, malformed, or token-limited tool-call response')
            calls = [FunctionCall(function=call['name'], args=call['arguments'], id=f'call_{len(generator.turns)}_{index}') for index, call in enumerate(calls)]

            # tool_calls already carries the JSON, so do not duplicate it in the next prompt
            assistant = {'role': 'assistant', 'content': [text_content_block_from_string('' if calls else result['text'])], 'tool_calls': calls}
            return query, runtime, env, [*messages, assistant], extra_args

    llm = LocalLLM()

    # tool execution adds an observation to the conversation, then the model responds again
    # continue this tool/model cycle up to the benchmark's configured loop limit
    loop = ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=15)
    pipeline = AgentPipeline([SystemMessage(load_system_message('default')), InitQuery(), llm, loop])
    pipeline.name = 'Qwen2.5'
    return pipeline


def run_dojo(case, suites, pipeline):
    Logger = import_module('agentdojo.logging').Logger

    class Capture(Logger):
        def __init__(self):
            self.messages = []

        def log(self, messages, **kwargs):
            self.messages = messages

        def log_error(self, message):
            pass

    suite = suites[case['suite']]
    injection = suite.injection_tasks[case['injection_task']] if case['injection_task'] else None

    # the upstream environment executes simulated tools and checks the resulting task and attack goals
    with Capture() as trace:
        utility, success = suite.run_task_with_pipeline(pipeline, suite.user_tasks[case['task']], injection, case['injections'])
    messages = json.loads(json.dumps(trace.messages, default=lambda value: value.model_dump()))
    return {'native_success': bool(success) if injection else bool(utility), 'task_success': bool(utility), 'trace': messages, 'tool_errors': sum(bool(message.get('error')) for message in trace.messages)}


def injecagent_messages(item, tools, prompts, stage2=False):
    names = [item['User Tool'], *item['Attacker Tools']]
    scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"
    if stage2:
        scratchpad += f"Thought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    text = prompts.USER_PROMPT.format(available_tool_descriptions=json.dumps([tools[name] for name in names], indent=True), input=item['User Instruction'], available_tool_names=names, agent_scratchpad=scratchpad)
    return [{'role': 'system', 'content': prompts.SYS_PROMPT}, {'role': 'user', 'content': text}]


def run_injecagent(case, generator, resources):

    prompts, checker, simulator, tools, saved = resources
    item = dict(case['item'])
    response = generator.generate(injecagent_messages(item, tools, prompts))

    # the original benchmark checker inspects whether the first response requests the attacker's tool
    first = checker.evaluate_output_prompted(response['text'], item['Attacker Tools'][0], item['Attacker Tools'], item['User Tool'], item['Attacker Instruction'])
    result = {'stage1': first, 'native_success': None if first['eval'] == 'invalid' else first['eval'] == 'succ', 'task_success': None}

    # data-theft cases need a second model response after retrieving the requested information
    if case['kind'] != 'ds' or first['eval'] != 'succ':
        return result
    item.update({key: first[key] for key in ['Attacker Tool Parameters', 'Attacker Tool Thought']})
    tool = item['Attacker Tools'][0]
    key = f"({tool},{item['Attacker Tool Parameters']})"
    result['simulator_source'] = 'published_cache'
    observation = saved.get(key)
    if observation is None:
        result['simulator_source'] = 'frozen_base_qwen_local'
        cache = OUTPUT / 'InjecAgent' / 'local_simulator' / (hashlib.sha256(key.encode()).hexdigest() + '.json')
        if cache.exists():
            simulated = json.loads(cache.read_text())
        else:
            prompt = simulator.DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE.format(example=simulator.EXAMPLE, attacker_tool=json.dumps(tools[tool], indent=True), tool_parameters=item['Attacker Tool Parameters'])
            simulated = generator.generate([{'role': 'system', 'content': simulator.SYSTEM_MESSAGE}, {'role': 'user', 'content': prompt}], base=True)
            cache.parent.mkdir(exist_ok=True)
            write_json(cache, simulated)
        blocks = re.findall(r'```(.*?)```', simulated['text'], re.DOTALL)
        observation = blocks[0].strip().removeprefix('json').strip() if blocks else ''
        if not observation or not simulated['finished']:
            result.update(native_success=None, error='local_simulator_did_not_return_a_complete_fenced_response')
            return result

    # return the simulated tool observation to the model before checking the final attacker action
    item['Attacker Tool Response'] = observation
    response = generator.generate(injecagent_messages(item, tools, prompts, stage2=True))
    second = checker.evaluate_output_prompted(response['text'], item['Attacker Tools'][1], item['Attacker Tools'], item['User Tool'], item['Attacker Instruction'])
    result.update(stage2=second, simulated_observation=observation, native_success=None if second['eval'] == 'invalid' else second['eval'] == 'succ')
    return result


def run(benchmark, method, smoke=False):
    folder = OUTPUT / benchmark
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, checksum in manifest['input_hashes'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f'Frozen benchmark input changed: {name}')
    cases = json.loads((folder / 'cases.json').read_text())
    if hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest() != manifest['cases_sha256']:
        raise ValueError('The frozen cases changed')
    if smoke:
        # one clean and one injection case per suite checks the complete tool loop
        groups = {}
        for case in cases:
            groups.setdefault((case.get('suite', case.get('kind')), case['condition']), case)
        cases = list(groups.values())
    folder = folder / ('smoke' if smoke else 'evaluation') / method
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'run.json').exists():
        old = json.loads((folder / 'run.json').read_text())
        if old['manifest_sha256'] != digest(OUTPUT / benchmark / 'manifest.json'):
            raise ValueError('Cannot mix results from different benchmark versions')
    previous = [json.loads(file.read_text()) for file in sorted(folder.glob('case_*.json'))]
    completed = {row['id'] for row in previous}
    if all(case['id'] in completed for case in cases):
        print('All requested cases already have results')
        return
    generator = LocalModel(method)
    if (folder / 'run.json').exists() and old['model'] != generator.metadata:
        raise ValueError('Cannot resume with a different adapter')
    if benchmark in ['AgentDojo', 'AgentDyn']:
        suites = load_dojo(benchmark)
        pipeline = dojo_pipeline(benchmark, generator)
    else:
        source = UPSTREAM / 'InjecAgent'
        prompts = import_file('injecagent_prompts', source / 'src/prompts/agent_prompts.py')
        checker = import_file('injecagent_checker', source / 'src/output_parsing.py')
        simulator = import_file('injecagent_simulator', source / 'src/prompts/generation_prompts.py')
        tools = {}
        for toolkit in json.loads((source / 'data/tools.json').read_text()):
            for tool in toolkit['tools']:
                tool['name'] = toolkit['toolkit'] + tool['name']
                tools[tool['name']] = tool
        saved = json.loads((source / 'data/attacker_simulated_responses.json').read_text())
        resources = prompts, checker, simulator, tools, saved
    block_network()
    status = {'status': 'running', 'benchmark': benchmark, 'total': len(cases), 'completed': len(previous), 'started_unix': time.time(), 'model': generator.metadata, 'manifest_sha256': digest(OUTPUT / benchmark / 'manifest.json')}
    write_json(folder / 'run.json', status)
    for index, case in enumerate(cases):
        if case['id'] in completed:
            continue
        generator.turns = []
        row = {**case, 'native_success': None, 'task_success': None}
        started = time.time()
        try:
            row.update(run_dojo(case, suites, pipeline) if benchmark != 'InjecAgent' else run_injecagent(case, generator, resources))
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
            if 'out of memory' in str(error).lower():
                raise
        row.update(turns=generator.turns, repetition=any(turn['repetitive_at_limit'] for turn in generator.turns), unfinished=any(not turn['finished'] for turn in generator.turns), elapsed_seconds=time.time() - started)

        # complete each record atomically so interrupted runs can resume safely
        temporary = folder / f'case_{index:05d}.tmp'
        write_json(temporary, row)
        temporary.replace(folder / f'case_{index:05d}.json')
        previous.append(row)
        status['completed'] = len(previous)
        write_json(folder / 'run.json', status)
        write_json(folder / 'metrics.json', summarize(previous))
        print(f'{benchmark} {method}: {len(previous)}/{len(cases)} {case["id"]} success={row["native_success"]} error={row.get("error")}', flush=True)
    status['status'] = 'complete'
    write_json(folder / 'run.json', status)
    if benchmark == 'InjecAgent':
        write_json(folder / 'metrics_by_setting.json', {setting: summarize([row for row in previous if row['setting'] == setting]) for setting in ['base', 'enhanced']})
    if smoke and any(row.get('error') for row in previous):
        raise RuntimeError('Smoke tests contain errors; inspect them before full evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['AgentDojo', 'AgentDyn', 'InjecAgent'], required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--method-index', type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.benchmark)
    else:
        run(args.benchmark, METHODS[args.method_index], args.smoke)
