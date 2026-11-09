"""Shared model generation and records for task evaluations"""

from pathlib import Path
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time

from peft import PeftModel
import torch

from scripts.curation.common import ROOT, digest, load_model, read_jsonl, write_json
from scripts.curation.evaluation.evaluate import repetition_check


# append dependencies so they cannot replace the existing PyTorch or Transformers
if os.environ.get('BENCHMARK_DEPS'):
    sys.path.append(os.environ['BENCHMARK_DEPS'])


UPSTREAM = ROOT / 'data/task_benchmarks/upstream'
OUTPUT = ROOT / 'model/artifacts/curation/qwen25_7b_external_tasks_v1'
METHODS = ['base', 'random', 'text_diversity', 'model_failures', 'logistic_regression', 'linear_svm', 'neural_network', 'jlens_llmail', 'jlens_synthetic']
CONFIG = ROOT / 'scripts/curation/configs/config_reference_groups.json'
TOOL_PARSER_VERSION = 'qwen_tagged_calls_v2'
COMMITS = {'AgentDojo': '089ed468cf3ed0322acc66b0211f26d9d90dbf60', 'AgentDyn': '5353cf7615b135cace8d07c8f12dac53a16b6db3', 'InjecAgent': 'f19c9f2c79a41046eb13c03c51a24c567a8ffa07'}


def import_file(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block_network():
    # benchmark tools must operate on simulated data, never real services
    def audit(event, args):
        if event == 'socket.connect' and args[0].family in [socket.AF_INET, socket.AF_INET6]:
            raise RuntimeError('Network calls are disabled during benchmark evaluation')
    sys.addaudithook(audit)


def quoted_regions(text):
    # fenced examples and reasoning blocks cannot dispatch tools
    spans = [(m.start(), m.end()) for m in re.finditer(r'<think>.*?(?:</think>|\Z)', text, re.S)]
    fence = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^[ \t]{0,3}(`{3,}|~{3,})', line)
        if marker:
            delimiter = marker.group(1)
            if fence is None:
                fence = (delimiter[0], len(delimiter), offset)
            elif delimiter[0] == fence[0] and len(delimiter) >= fence[1] and not line[marker.end():].strip():
                spans.append((fence[2], offset + len(line)))
                fence = None
        if line.lstrip().startswith('>'):
            spans.append((offset, offset + len(line)))
        offset += len(line)
    if fence is not None:
        spans.append((fence[2], len(text)))

    # matching backtick runs also protect inline and multiline code quotes
    spans.extend((m.start(), m.end()) for m in re.finditer(r'(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)', text, re.S))
    return spans


def unique_json_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'Duplicate JSON field: {name}')
        result[name] = value
    return result


def invalid_json_constant(value):
    raise ValueError(f'Invalid JSON constant: {value}')


def parse_tool_calls(text, *, finished=True):
    """Read explicit Qwen calls outside quoted examples; return calls and a format-error flag"""
    if not finished:
        return [], '<tool_call>' in text
    spans = quoted_regions(text)
    quote_text = text

    # keep only requests outside quoted examples and generated reasoning blocks
    calls = []
    ambiguous = False
    decoder = json.JSONDecoder(object_pairs_hook=unique_json_fields, parse_constant=invalid_json_constant)
    cursor = 0
    previous_end = None
    while (start := text.find('<tool_call>', cursor)) >= 0:
        cursor = start + len('<tool_call>')
        if any(left <= start < right for left, right in spans):
            continue

        # a call can follow prose on earlier lines, but inline quoted mentions stay text
        line_start = text.rfind('\n', 0, start) + 1
        after_previous = previous_end is not None and not text[previous_end:start].strip()
        if text[line_start:start].strip() and not after_previous:
            # unclear inline markup stays unscored instead of becoming a silent no-call result
            ambiguous = True
            continue
        payload_start = cursor + len(text[cursor:]) - len(text[cursor:].lstrip())
        try:
            call, end = decoder.raw_decode(text, payload_start)
        except ValueError:
            return [], True
        if not isinstance(call, dict) or not isinstance(call.get('name'), str) or not call['name'].strip() or not isinstance(call.get('arguments'), dict):
            return [], True
        tail = end + len(text[end:]) - len(text[end:].lstrip())
        if text.startswith('</tool_call>', tail):
            end = tail + len('</tool_call>')
        else:
            # retain complete calls with an omitted closing tag, but reject garbage inside a closed block
            closing = text.find('</tool_call>', end)
            following = text.find('<tool_call>', end)
            if closing >= 0 and (following < 0 or closing < following):
                return [], True
        calls.append(call)

        # text inside JSON arguments cannot open a reasoning block or code quote around later calls
        quote_text = quote_text[:payload_start] + re.sub(r'[^\n]', ' ', quote_text[payload_start:end]) + quote_text[end:]
        spans = quoted_regions(quote_text)
        cursor = previous_end = end
    return ([], True) if ambiguous else (calls, False)


def freeze_manifest(benchmark, rows, details):

    source = UPSTREAM / benchmark
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMITS[benchmark]:
        raise ValueError(f'Unexpected source version for {benchmark}')
    files = [file for file in source.rglob('*') if file.is_file() and '.git' not in file.parts and '__pycache__' not in file.parts]
    files += [CONFIG, Path(__file__), ROOT / 'scripts/curation/evaluation/evaluate_task_benchmarks.py']

    # bind every adapter comparison to the same prepared cases, source version and generation limits
    case_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    manifest = {'benchmark': benchmark, 'source_commit': commit, 'methods': METHODS, 'case_count': len(rows), 'cases_sha256': case_hash, 'generation': {'dtype': 'bfloat16', 'seed': 42, 'batch_size': 1, 'do_sample': False, 'max_new_tokens': 1024, 'max_context_tokens': 32768}, 'paid_apis': False, 'training': False, 'details': details, 'input_hashes': {str(file.relative_to(ROOT)): digest(file) for file in sorted(files)}}
    folder = OUTPUT / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest and any(folder.rglob('case_*.json')):
        raise ValueError('The frozen benchmark changed; use a new output version')
    write_json(target, manifest)
    cases = folder / 'cases.json'
    if cases.exists() and json.loads(cases.read_text()) != rows and any(folder.rglob('case_*.json')):
        raise ValueError('The frozen cases changed')
    write_json(cases, rows)
    print(f'{benchmark}: {len(rows)} cases prepared', flush=True)


class LocalModel:
    def __init__(self, method):

        torch.set_num_threads(4)
        torch.manual_seed(42)
        config = json.loads(CONFIG.read_text())
        self.model, self.tokenizer = load_model(config)
        self.metadata = {'method': method, 'model': config['model'], 'config_sha256': digest(CONFIG)}
        if method != 'base':
            folder = ROOT / config['output'] / 'adapters' / method
            training = json.loads((folder / 'training_run.json').read_text())
            selected = json.loads((folder / 'selected_checkpoint.json').read_text())
            if training['status'] != 'complete' or training['config_sha256'] != digest(CONFIG):
                raise ValueError('The adapter does not match the completed approved training')
            checkpoint = folder / selected['path']
            self.model = PeftModel.from_pretrained(self.model, checkpoint, is_trainable=False)
            self.metadata.update(checkpoint=str(checkpoint), selected_epoch=selected['epoch'], adapter_sha256=digest(checkpoint / 'adapter_model.safetensors'))
        self.model.eval()
        self.model.config.use_cache = True
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.turns = []

    def generate(self, messages, tools=None, base=False):

        text = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
        length = inputs.input_ids.shape[1]
        if length + 1024 > min(32768, self.model.config.max_position_embeddings):
            raise ValueError(f'context_limit: {length} input tokens, no prompt truncation')
        manager = self.model.disable_adapter() if base and hasattr(self.model, 'disable_adapter') else contextlib.nullcontext()
        with manager, torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=1024, pad_token_id=self.tokenizer.pad_token_id)
        ids = generated[0, length:].cpu().tolist()
        result = {'messages': messages, 'tools': tools, 'input_tokens': length, 'response_ids': ids, 'generated_tokens': len(ids), 'finished': bool(ids and ids[-1] in self.eos), 'text': self.tokenizer.decode(ids, skip_special_tokens=True)}
        result.update(repetition_check(result, 1024))
        if not base:
            self.turns.append(result)
        return result


def summarize(rows):
    summary = {}
    for condition in ['safe', 'injection']:
        group = [row for row in rows if row['condition'] == condition]
        scored = [row for row in group if row.get('native_success') is not None]
        utility = [row for row in group if row.get('task_success') is not None]
        summary[condition] = {'cases': len(group), 'scored_cases': len(scored), 'errors_or_unscored': len(group) - len(scored), 'native_success_rate': sum(row['native_success'] for row in scored) / len(scored) if scored else None, 'task_completion_rate': sum(row['task_success'] for row in utility) / len(utility) if utility else None, 'repetitive_cases': sum(row.get('repetition', False) for row in group), 'unfinished_cases': sum(row.get('unfinished', False) for row in group)}
        if condition == 'injection':
            known = [row for row in group if row.get('native_success') is not None or row.get('repetition')]
            summary[condition]['success_or_repetition_rate'] = sum(bool(row.get('native_success')) or row.get('repetition', False) for row in known) / len(known) if known else None
            summary[condition]['success_or_repetition_denominator'] = len(known)
    return summary
