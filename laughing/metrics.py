"""
Metrics for evaluation of unit test generation capability.
"""
import subprocess
import re
import json
import os

def split_into_functions(text):
    """Split the text containing multiple functions into individual functions."""
    lines = text.split('\n')
    functions = []
    function = ''
    for line in lines:
        if line.strip().startswith('def '):
            if function.strip().startswith('def '):
                functions.append(function)
            function = line + '\n'
        else:
            function += line + '\n'
    functions.append(function)

    return functions

def check_compile_errors(functions):
    """Check the functions for complie error.
    Also return the functions that can be compiled.
    """
    error_count = 0
    no_error_functions = []

    for function in functions:
        try:
            compile(function, '<string>', 'exec')
            no_error_functions.append(function)
        except SyntaxError:
            error_count += 1

    return error_count, no_error_functions

def eval_unit_tests(program: str, test_cases: str, cov_output_file:str ='cov.json'):
    """Evaluate unit test cases of a single program.

    Given a program and its test cases, calculate the metrics for unit tests.
    """
    # first get compiler errors
    functions = split_into_functions(test_cases)
    compile_errors, valid_funcs = check_compile_errors(functions)
    metrics = {
        'compile_errors': compile_errors, 'failed': 0, 'passed': 0
    }

    with open('program.py', mode='w') as fout:
      fout.write(program)

    with open('test_cases.py', mode='w') as fout:
      fout.write('from program import *\n')
      fout.writelines(valid_funcs)

    result = subprocess.run(['coverage', 'run', '--branch', '-m', 'pytest', 'test_cases.py'],
                            stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    matching = re.search(r'(\d+) failed', output)
    if matching:
        metrics['failed'] = int(matching.group(1))
    matching = re.search(r'(\d+) passed', output)
    if matching:
        metrics['passed'] = int(matching.group(1))

    subprocess.run(['coverage', 'json', "--include=program.py", '-o', cov_output_file])
    with open(cov_output_file) as fout:
        cov = json.load(fout)['totals']
        metrics['stm_covered_pct'] = cov['covered_lines']*1.0 / cov['num_statements']
        metrics['branch_covered_pct'] = (cov['covered_branches']*1.0 / cov['num_branches']
                                         if cov['num_branches'] > 0 else 1.0)
        try:
          os.remove(cov_output_file)
        except Exception as ex:
          print(ex)

    return metrics
